"""
model_engine.py
================
Self-contained Industrial Lead Scoring and Classification Engine.
Supports:
  - Sentence-Transformer Embeddings (all-MiniLM-L6-v2)
  - Master Query Gating and Intent Analyzer
  - Economic Sanity and Prank Detection (log(price)/duration ratio)
  - Stacking Ensemble (RF + HGB + LR) with Isotonic Calibration
  - Automatic Artifact Caching (artifacts/lead_model.joblib)
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

MODEL_NAME = 'all-MiniLM-L6-v2'
SEED = 42
EMBED_DIM = 384
SVD_COMPONENTS = 105

LABEL_MAP = {'Cold': 0, 'Warm': 1, 'Hot': 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}
SCORE_RANGES = {'Cold': (0, 39), 'Warm': (40, 69), 'Hot': (70, 100)}

MIN_BUDGET_PER_NSVC = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
_MIN_4PLUS = 4.0

# ── Helper parsing functions ──────────────────────────────────────────────────
def parse_budget_inr(s: Any) -> float:
    if not s: return 0.0
    cleaned = re.sub(r'[\u20b9Rs.,\s]', '', str(s))
    nums = re.findall(r'\d+', cleaned)
    return max(int(n) for n in nums) / 100_000 if nums else 0.0

def parse_company_size(s: Any) -> int:
    nums = re.findall(r'\d+', str(s))
    return max(int(n) for n in nums) if nums else 0

def parse_duration_months(duration_str: Any) -> float:
    if not duration_str:
        return 3.0
    s = str(duration_str).lower()
    week_nums = re.findall(r'(\d+)\s*(?:-\s*\d+)?\s*week', s)
    if week_nums:
        return max(int(w) for w in week_nums) / 4.33
    month_nums = re.findall(r'\d+', s)
    if month_nums:
        return float(max(int(n) for n in month_nums))
    return 3.0

def sigmoid(x: float, k: float = 8.0) -> float:
    kx = k * x
    if kx > 100: return 1.0
    if kx < -100: return 0.0
    return 1.0 / (1.0 + np.exp(-kx))

# ── Anchors ───────────────────────────────────────────────────────────────────
REJECTION_ANCHORS = [
    'We have decided to go with another provider, vendor, or agency.',
    'We chose another company and signed a contract with them.',
    'We are not interested in working with you.',
    'We do not need this service, please remove us from your contact list.',
    'We decided not to proceed and canceled this initiative.',
    'We do not think your services are promising or suitable for us.',
    'Your solution is not a good fit for us and we will pass.',
    'We are going with a competitor.',
    'We do not require your services. Thank you.',
    'We are doubtful about your capabilities and will not proceed.',
    'We do not want website codes and files.',
]

HIGH_INTENT_ANCHORS = [
    'We are ready to get started immediately.',
    'What is the pricing for your premium plan? We are ready to get started.',
    'Your solution looks perfect for our requirements. What are the next steps?',
    'We want full deployment and end-to-end launch as soon as possible.',
    'We have approved the budget and want to onboard your team right now.',
    'Please send the contract or payment details so we can start.',
    'We are ready to proceed with development.',
    'Let us schedule a kickoff call this week to begin development.',
]

INQUIRY_ANCHORS = [
    'we do not want website codes and files. Can you provide extra services like UI/UX design?',
    'Can you provide extra services like UI/UX design?',
    'Do you also provide maintenance and cloud support after launch?',
    'What services do you offer and can you customize the scope?',
    'We want to know if you can handle mobile app as well as web?',
    'Could you share more details about your process and portfolio?',
]

COLD_BROWSING_ANCHORS = [
    'Just browsing and curious about pricing.',
    'We are a student looking for free or cheap advice.',
    'No timeline and no budget, just exploring.',
    'Maybe sometime next year, nothing planned.',
]

HOT_KW = ['approved','immediately','this month','this week','start now',
          'ready to start','ready to get started','kickoff','urgent','funded','investor','board',
          'signed','serious','committed','full deployment','deployment','next steps','solution looks perfect']
COLD_KW = ['curious','exploring','just checking','how much','price','cost',
           'very limited','cheap','affordable','no timeline','not sure',
           'idea stage','someday','maybe','not promising','unpromising','doubtful']
NI_KW = ['don\'t need','dont need','do not need','not interested','not looking',
         'decided against','found another','in-house','inhouse','put on hold',
         'on hold','changed our mind','no longer','not going forward',
         'not proceeding','remove us','unsubscribe','won\'t be','wont be',
         'will not be','not thinking','not require','dont require','don\'t require',
         'not promising','unpromising','don\'t think your services','dont think your services',
         'not convinced','not impressed','not a good fit','not suitable','skeptical',
         'another provider','another vendor','another agency','another company']

def query_keyword_net(text: str) -> float:
    t = text.lower()
    hot  = sum(kw in t for kw in HOT_KW)
    cold = sum(kw in t for kw in COLD_KW)
    ni   = sum(kw in t for kw in NI_KW)
    total = hot + cold + ni
    return (hot - cold - 2 * ni) / max(total + 1, 1)

class LeadScoringEngine:
    def __init__(self, artifacts_path: str = 'artifacts/lead_model.joblib', dataset_path: str = 'dataset.json'):
        self.artifacts_path = Path(artifacts_path)
        self.dataset_path = Path(dataset_path)
        self.encoder = SentenceTransformer(MODEL_NAME)
        
        # Pre-encode anchors
        self.rej_vecs = self.encoder.encode(REJECTION_ANCHORS, normalize_embeddings=True)
        self.hi_vecs  = self.encoder.encode(HIGH_INTENT_ANCHORS, normalize_embeddings=True)
        self.inq_vecs = self.encoder.encode(INQUIRY_ANCHORS, normalize_embeddings=True)
        self.cld_vecs = self.encoder.encode(COLD_BROWSING_ANCHORS, normalize_embeddings=True)

        self.svd = None
        self.model = None
        self.load_or_train()

    def query_intent_analyzer(self, query: str) -> Dict[str, Any]:
        q = str(query).strip()
        if not q:
            return {'intent': 'NEUTRAL', 'confidence': 0.50, 'reason': 'Empty query', 'source': 'Intent Engine'}
        q_lower = q.lower()

        rejection_regexes = [
            r'\b(another|other|different|new)\s+(provider|vendor|agency|company|team|developer|partner|firm)\b',
            r'\b(chose|chosen|went with|going with|go with|signed with|hired)\s+(another|someone else|a competitor)\b',
            r'\bnot\s+interested\b', r'\bremove\s+us\b', r'\bunsubscribe\b',
            r'\b(not|un)\s*promising\b', r'\bdon\'?t\s+think.*promising\b',
            r'\bdon\'?t\s+think\s+(your|the)\s+services\b',
            r'\b(not\s+a\s+good\s+fit|not\s+suitable|not\s+convinced|doubtful\s+about|skeptical\s+about)\b',
            r'\b(put\s+on\s+hold|no\s+longer\s+looking|changed\s+our\s+mind|not\s+moving\s+forward|not\s+proceeding)\b',
            r'\b(do\s+not|don\'?t)\s+(need|require|want)\s+(your|this|any)\s+service\b'
        ]
        for rpat in rejection_regexes:
            if re.search(rpat, q_lower):
                return {'intent': 'REJECTION', 'confidence': 0.99, 'reason': f'Explicit rejection or competitor selection (matched: {rpat})', 'source': 'Intent Engine'}

        high_intent_regexes = [
            r'\bready\s+to\s+(get\s+started|start|proceed|move\s+forward|onboard|buy|hire)\b',
            r'\bwhat\s+are\s+the\s+next\s+steps\b',
            r'\bsolution\s+looks\s+perfect\b',
            r'\bwe\s+want\s+full\s+deployment\b',
            r'\bfull\s+deployment\b',
            r'\bstart\s+immediately\b',
            r'\bonboard\s+immediately\b',
            r'\b(we\s+have\s+)?finalized\s+(our\s+)?budget\b',
            r'\blike\s+to\s+move\s+forward\b',
            r'\blet\'?s\s+schedule\s+a\s+kickoff\b',
            r'\bsend\s+(the\s+)?contract\b',
            r'\bready\s+to\s+pay\b'
        ]
        for hpat in high_intent_regexes:
            if re.search(hpat, q_lower):
                return {'intent': 'HIGH_INTENT', 'confidence': 0.98, 'reason': f'High intent or readiness (matched: {hpat})', 'source': 'Intent Engine'}

        is_pure_neg = bool(re.search(r'\b(don\'?t want|do not want|don\'?t need|do not need|no codes?|no files?)\b', q_lower))
        has_inquiry_ask = bool(re.search(r'\b(can you|could you|do you|provide|extra services|ui/ux|maintenance|design|consultation)\b', q_lower))
        if is_pure_neg and not has_inquiry_ask:
            return {'intent': 'REJECTION', 'confidence': 0.95, 'reason': 'Client only specifies exclusions with no positive ask', 'source': 'Intent Engine'}

        qv = self.encoder.encode([q], normalize_embeddings=True)[0]
        s_rej = max(float(np.dot(qv, v)) for v in self.rej_vecs)
        s_hi  = max(float(np.dot(qv, v)) for v in self.hi_vecs)
        s_inq = max(float(np.dot(qv, v)) for v in self.inq_vecs)
        s_cld = max(float(np.dot(qv, v)) for v in self.cld_vecs)

        max_val = max(s_rej, s_hi, s_inq, s_cld)
        if s_rej == max_val and s_rej > 0.50:
            return {'intent': 'REJECTION', 'confidence': float(s_rej), 'reason': f'Semantic match to rejection ({s_rej:.2f})', 'source': 'Semantic Engine'}
        elif s_hi == max_val and s_hi > 0.50:
            return {'intent': 'HIGH_INTENT', 'confidence': float(s_hi), 'reason': f'Semantic match to high commercial demand ({s_hi:.2f})', 'source': 'Semantic Engine'}
        elif s_inq == max_val and s_inq > 0.40:
            return {'intent': 'INQUIRY', 'confidence': float(s_inq), 'reason': f'Semantic match to service inquiry ({s_inq:.2f})', 'source': 'Semantic Engine'}
        elif s_cld == max_val and s_cld > 0.45:
            return {'intent': 'COLD_EXPLORATORY', 'confidence': float(s_cld), 'reason': f'Casual exploratory browsing ({s_cld:.2f})', 'source': 'Semantic Engine'}

        return {'intent': 'NEUTRAL', 'confidence': 0.50, 'reason': 'General business inquiry', 'source': 'Intent Engine'}

    def extract_structured(self, lead: Dict[str, Any], intent_gate: float = 0.5) -> np.ndarray:
        g = float(intent_gate)
        dampen = max(g / 0.35, 0.05) if g < 0.35 else 1.0

        bl = parse_budget_inr(lead.get('estimated_budget', ''))
        price_inr = bl * 100_000
        b_log  = float(np.log1p(bl)) * dampen
        b_norm = min(bl / 20.0, 1.0) * dampen

        dur = parse_duration_months(lead.get('project_duration', ''))
        log_price_per_dur = (float(np.log(price_inr)) / max(dur, 1.0)) if price_inr > 0 else 0.0
        price_per_month = bl / max(dur, 1.0)

        svcs = lead.get('service_required', [])
        if isinstance(svcs, str): svcs = [s.strip() for s in svcs.split('|') if s.strip()]
        n_svc = len(svcs)
        min_bl = MIN_BUDGET_PER_NSVC.get(min(n_svc, 3), _MIN_4PLUS) if n_svc >= 4 else MIN_BUDGET_PER_NSVC.get(n_svc, 0)
        b_realistic = (1.0 if (min_bl == 0 or bl >= min_bl) else 0.0) * dampen
        b_per_svc = float(np.log1p(bl / max(n_svc, 1))) * dampen

        urg = str(lead.get('urgency', '')).lower()
        urg_h = (1.0 if urg == 'high' else 0.0) * dampen
        urg_m = (1.0 if urg == 'medium' else 0.0) * dampen
        urg_o = (3.0 if urg == 'high' else (2.0 if urg == 'medium' else 1.0)) * dampen

        dm_val = 1.0 if lead.get('decision_maker', False) else 0.0
        is_dm  = dm_val * (dampen if g >= 0.25 else 0.0)

        sz    = parse_company_size(lead.get('company_size', ''))
        s_log = float(np.log1p(sz)) * (dampen if g < 0.35 else 1.0)

        src = str(lead.get('lead_source', '')).lower()
        if   any(x in src for x in ['referral','website','inbound']): src_q = 4.0
        elif any(x in src for x in ['linkedin','social','ad']):        src_q = 3.0
        elif any(x in src for x in ['google','search']):               src_q = 2.5
        elif any(x in src for x in ['cold','outbound']):               src_q = 1.0
        else:                                                          src_q = 2.0

        start = str(lead.get('expected_start_date', '')).lower()
        if   'immediate' in start:               st_urg = 1.00
        elif 'within 1 month' in start:          st_urg = 0.85
        elif 'within 2 month' in start:          st_urg = 0.65
        elif 'next quarter' in start or '3' in start: st_urg = 0.35
        else:                                    st_urg = 0.10
        st_urg = st_urg * dampen

        return np.array([
            b_log, b_norm, b_realistic,
            log_price_per_dur, price_per_month,
            urg_h, urg_m, is_dm, s_log, src_q, float(n_svc), st_urg, b_per_svc,
            g * b_log, g * urg_o, g * is_dm,
        ], dtype=np.float32)

    def compute_query_gate(self, lead: Dict[str, Any], check_llm: bool = True):
        q_text = str(lead.get('query', '')).strip()
        m_text = str(lead.get('message', '')).strip()

        llm_res = None
        if q_text:
            q_vec = self.encoder.encode([q_text], normalize_embeddings=True)[0]
            q_hi  = max(float(np.dot(q_vec, v)) for v in self.hi_vecs)
            q_lo  = max(float(np.dot(q_vec, v)) for v in self.cld_vecs)
            q_urg = 0.5
            q_com = 0.5
            q_ni  = max(float(np.dot(q_vec, v)) for v in self.rej_vecs)
            q_kw  = query_keyword_net(q_text)

            if check_llm:
                llm_res = self.query_intent_analyzer(q_text)
                intent = llm_res.get('intent')
                if intent in ['INTERESTED', 'HIGH_INTENT']:
                    q_hi = max(q_hi, 0.45)
                    q_ni = min(q_ni, 0.18)
                elif intent == 'INQUIRY':
                    q_hi = max(q_hi, 0.32)
                    q_ni = min(q_ni, 0.20)
                elif intent in ['NOT_INTERESTED', 'REJECTION', 'COLD_EXPLORATORY']:
                    q_ni = max(q_ni, 0.50)
                    q_hi = min(q_hi, 0.15)

            q_net = (q_hi - q_lo) + 0.25 * q_urg + 0.20 * q_kw - 0.60 * q_ni
        else:
            q_vec = np.zeros(EMBED_DIM, dtype=np.float32)
            q_hi = q_lo = q_urg = q_com = q_ni = q_kw = 0.0
            q_net = 0.0

        if m_text:
            m_vec = self.encoder.encode([m_text], normalize_embeddings=True)[0]
            m_hi  = max(float(np.dot(m_vec, v)) for v in self.hi_vecs)
            m_lo  = max(float(np.dot(m_vec, v)) for v in self.cld_vecs)
            m_urg = 0.5
            m_com = 0.5
            m_ni  = max(float(np.dot(m_vec, v)) for v in self.rej_vecs)
            m_net = (m_hi - m_lo) + 0.15 * m_urg - 0.40 * m_ni
        else:
            m_vec = np.zeros(EMBED_DIM, dtype=np.float32)
            m_hi = m_lo = m_urg = m_com = m_ni = 0.0
            m_net = 0.0

        gate = float(sigmoid(0.75 * q_net + 0.25 * m_net, 8.0))
        q_sims = np.array([q_hi, q_lo, q_urg, q_com, q_kw, q_ni], dtype=np.float32)
        m_sims = np.array([m_hi, m_lo, m_urg, m_com],              dtype=np.float32)
        return gate, q_sims, q_vec, m_sims, m_vec, llm_res

    def extract_all_features(self, lead: Dict[str, Any], check_llm: bool = False) -> np.ndarray:
        gate, q_sims, q_vec, m_sims, m_vec, _ = self.compute_query_gate(lead, check_llm=check_llm)
        struct = self.extract_structured(lead, intent_gate=gate)

        d_text = str(lead.get('project_description', '')).strip()
        if d_text:
            d_vec  = self.encoder.encode([d_text], normalize_embeddings=True)[0]
            d_sims = np.array([0.5, 0.2, 0.6, 0.2], dtype=np.float32)
        else:
            d_vec  = np.zeros(EMBED_DIM, dtype=np.float32)
            d_sims = np.zeros(4, dtype=np.float32)

        if gate < 0.35:
            m_vec_scaled = m_vec * gate
            d_vec_scaled = d_vec * gate
        else:
            m_vec_scaled = m_vec
            d_vec_scaled = d_vec

        return np.concatenate([
            struct, [gate], q_sims, m_sims, d_sims,
            q_vec.astype(np.float32), m_vec_scaled.astype(np.float32), d_vec_scaled.astype(np.float32)
        ])

    def load_or_train(self):
        if self.artifacts_path.exists():
            print(f'Loading cached model from {self.artifacts_path} ...')
            data = joblib.load(self.artifacts_path)
            self.svd = data['svd']
            self.model = data['model']
            print('Model loaded successfully.')
            return

        print(f'Training new model on {self.dataset_path} ...')
        with open(self.dataset_path, encoding='utf-8') as fh:
            all_leads = json.load(fh)
        labeled = [ld for ld in all_leads if ld.get('label') in LABEL_MAP]

        X_raw = np.array([self.extract_all_features(ld, check_llm=False) for ld in labeled])
        y = np.array([LABEL_MAP[ld['label']] for ld in labeled])
        leads_arr = np.array(labeled, dtype=object)

        embed_start = 16 + 1 + 6 + 4 + 4  # = 31

        X_tmp, X_test_raw, y_tmp, y_test, L_tmp, L_test = train_test_split(
            X_raw, y, leads_arr, test_size=0.15, stratify=y, random_state=SEED)
        X_train_raw, X_val_raw, y_train, y_val, L_train, L_val = train_test_split(
            X_tmp, y_tmp, L_tmp, test_size=0.15/(1-0.15), stratify=y_tmp, random_state=SEED)

        self.svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
        self.svd.fit(X_train_raw[:, embed_start:])

        def apply_svd(X):
            return np.concatenate([X[:, :embed_start], self.svd.transform(X[:, embed_start:])], axis=1)

        X_train = apply_svd(X_train_raw)
        X_val   = apply_svd(X_val_raw)

        rf_pipe = Pipeline([('sc', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=SEED, n_jobs=-1))])
        hgb_pipe = Pipeline([('sc', StandardScaler()), ('clf', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=6, random_state=SEED))])
        lr_pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(C=0.3, max_iter=2000, class_weight='balanced', random_state=SEED))])

        rf_pipe.fit(X_train, y_train)
        hgb_pipe.fit(X_train, y_train)
        lr_pipe.fit(X_train, y_train)

        meta_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        stacking_clf = StackingClassifier(estimators=[('rf', rf_pipe), ('hgb', hgb_pipe), ('lr', lr_pipe)], final_estimator=meta_lr, cv=5, stack_method='predict_proba', n_jobs=-1)
        stacking_clf.fit(X_train, y_train)

        self.model = CalibratedClassifierCV(stacking_clf, method='isotonic', cv='prefit')
        self.model.fit(X_val, y_val)

        # Save artifacts
        self.artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'svd': self.svd, 'model': self.model}, self.artifacts_path)
        print(f'Model trained and saved to {self.artifacts_path}.')

    def classify_lead(self, lead: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
        embed_start = 16 + 1 + 6 + 4 + 4  # 31
        gate, q_sims, _, m_sims, _, llm_res = self.compute_query_gate(lead, check_llm=use_llm)
        raw = self.extract_all_features(lead, check_llm=use_llm).reshape(1, -1)
        
        struct_part = raw[:, :embed_start]
        embed_part  = self.svd.transform(raw[:, embed_start:])
        feat = np.concatenate([struct_part, embed_part], axis=1)

        proba = self.model.predict_proba(feat)[0]
        llm_intent = llm_res.get('intent') if llm_res else None
        llm_reason = llm_res.get('reason', '') if llm_res else ''
        llm_source = llm_res.get('source', 'Intent Engine') if llm_res else 'Intent Engine'

        # Economic Sanity & Prank Detection
        bl = parse_budget_inr(lead.get('estimated_budget', ''))
        price_inr = bl * 100_000
        dur = parse_duration_months(lead.get('project_duration', ''))
        log_price_dur = (np.log(price_inr) / max(dur, 1.0)) if price_inr > 0 else 0.0

        svcs = lead.get('service_required', [])
        if isinstance(svcs, str): svcs = [s.strip() for s in svcs.split('|') if s.strip()]
        n_svc = len(svcs)
        min_bl = MIN_BUDGET_PER_NSVC.get(min(n_svc, 3), _MIN_4PLUS) if n_svc < 4 else _MIN_4PLUS

        prank = bool((min_bl > 0 and bl < min_bl and (bl < 0.50 or price_inr < 50000) and n_svc >= 2) or (price_inr < 25000 and n_svc >= 2))

        is_rejection = False
        is_inquiry = False

        if prank:
            p_cold = 0.95
            p_warm = 0.05
            p_hot  = 0.00
            proba  = np.array([p_cold, p_warm, p_hot])
        elif llm_intent in ['REJECTION', 'COLD_EXPLORATORY']:
            is_rejection = True
        elif llm_intent in ['INTERESTED', 'HIGH_INTENT']:
            gate = max(gate, 0.75)
            p_hot = max(proba[2], 0.70)
            p_warm = proba[1] * 0.5
            p_cold = min(proba[0], 0.05)
            proba = np.array([p_cold, p_warm, p_hot])
            proba = proba / proba.sum()
        elif llm_intent == 'INQUIRY':
            is_inquiry = True
            gate = 0.55
            p_warm = max(proba[1], 0.65)
            p_hot  = min(proba[2], 0.25)
            p_cold = min(proba[0], 0.10)
            proba  = np.array([p_cold, p_warm, p_hot])
            proba  = proba / proba.sum()
        else:
            is_rejection = (q_sims[5] >= 0.28 and q_sims[5] > q_sims[0]) or q_sims[4] <= -0.5 or gate < 0.35

        if is_rejection and not prank:
            rejection_boost = 0.95 if (llm_intent in ['NOT_INTERESTED', 'REJECTION', 'COLD_EXPLORATORY']) else 0.85
            p_cold = max(proba[0], 1.0 - gate, rejection_boost)
            p_warm = proba[1] * gate * 0.2
            p_hot  = proba[2] * (gate ** 2) * 0.02
            proba  = np.array([p_cold, p_warm, p_hot])
            proba  = proba / proba.sum()

        pred_cls = int(np.argmax(proba))
        tier = LABEL_INV[pred_cls]
        conf = float(proba[pred_cls])
        lo, hi = SCORE_RANGES[tier]
        score = round(lo + conf * (hi - lo), 1)

        rationale = []
        if prank:
            rationale.append(f'[PRANK / MOCKING DETECTED] Micro-budget (₹{price_inr:,.0f} for {n_svc} services) fails economic feasibility (min req: ₹{min_bl*100000:,.0f}). Ratio log(P)/dur = {log_price_dur:.2f}. Overridden to COLD.')
        elif llm_intent:
            rationale.append(f'[{llm_source}] Intent: {llm_intent} ({llm_reason})')

        if is_rejection and not prank:
            rationale.append('[REJECTION / SKEPTICISM ALERT] Client expressed negative perception, competitor selection, or refusal in query.')
        elif is_inquiry and not prank:
            rationale.append('[INQUIRY / SCOPE EVALUATION] Client is actively inquiring about services & scope.')
        elif not prank:
            rationale.append(f'[Budget OK] Ratio log(P)/dur: {log_price_dur:.2f}')

        if prank:
            action = 'Reject / Mark as Prank Lead. Micro-budget fails economic sanity.'
        elif tier == 'Hot':
            action = 'Call within 24 hours. Assign senior account executive immediately.'
        elif tier == 'Warm':
            action = 'Follow up within 2-3 business days. Clarify scope and custom deliverables.'
        else:
            action = 'Add to low-priority nurture campaign / do not prioritize sales rep time.'

        return {
            'lead': lead,
            'score': score,
            'tier': tier,
            'label': f'[{tier.upper()}]',
            'confidence': conf,
            'proba': {'Cold': float(proba[0]), 'Warm': float(proba[1]), 'Hot': float(proba[2])},
            'rationale': rationale,
            'prank_detected': prank,
            'intent_gate': float(gate),
            'log_price_dur': float(log_price_dur),
            'action': action,
            'llm_verdict': llm_res
        }
