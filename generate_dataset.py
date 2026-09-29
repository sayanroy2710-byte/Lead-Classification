#!/usr/bin/env python3
"""
generate_dataset.py
===================
Generates a realistic synthetic dataset of 1,100 labeled sales leads.

Output : dataset.json  (array of lead dicts, each with a 'label' field)
Classes: Hot (~35%)  Warm (~35%)  Cold (~30%)

Usage  : python generate_dataset.py
"""

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ── Lead counts per tier ──────────────────────────────────────────────────────
N_HOT  = 385
N_WARM = 385
N_COLD = 330

# ── Geography ─────────────────────────────────────────────────────────────────
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Kochi",
    "Noida", "Gurugram", "Indore", "Nagpur", "Chandigarh", "Bhopal",
]

# ── Names ─────────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Rahul","Priya","Aditya","Sneha","Vikram","Anjali","Rohan","Meera",
    "Suresh","Neha","Kiran","Mohit","Ritu","Gaurav","Pooja","Amit",
    "Divya","Rajesh","Sunita","Arjun","Kavya","Deepak","Lakshmi",
    "Sanjay","Anita","Ravi","Preeti","Vijay","Nisha","Manoj","Swati",
    "Ashok","Rekha","Sunil","Geeta","Rakesh","Sapna","Naresh","Seema",
    "Ramesh","Shweta","Praveen","Jyoti","Harish","Vandana","Nitin",
    "Poonam","Girish","Sunita","Bhavesh","Sheetal","Dhruv","Shruti",
    "Akash","Kajal","Yash","Pallavi","Karan","Simran","Varun","Tanvi",
]
LAST_NAMES = [
    "Sharma","Mehta","Kapoor","Joshi","Nair","Singh","Das","Krishnan",
    "Pillai","Gupta","Tiwari","Kulkarni","Verma","Reddy","Patil","Roy",
    "Malhotra","Arora","Shah","Patel","Iyer","Banerjee","Chauhan","Mishra",
    "Pandey","Rao","Kumar","Chaudhary","Bose","Sen","Dubey","Yadav",
    "Tripathi","Bajaj","Shetty","Menon","Naik","Desai","Thakur","Saxena",
]

# ── Company name parts ────────────────────────────────────────────────────────
COMPANY_PREFIXES = [
    "TechEdge","NextGen","SwiftByte","ProCode","DataBridge","CloudNest",
    "PixelForge","InnoStar","DeepTech","AlphaWave","BetaLogic","GreenBit",
    "SkyRocket","ZenCode","BluePeak","NovaSys","QuantumApp","FutureCore",
    "SmartFlow","RapidBuild","AgileHub","VisionTech","PrimeSoft","EcoApps",
    "MetroTech","UrbanByte","GlobalEdge","NeoWave","SparkLogic","DigiNest",
]
COMPANY_SUFFIXES = [
    "Pvt. Ltd.","Technologies","Solutions","Ventures","Innovations",
    "Systems","Enterprises","Labs","Digital","Consulting","Services",
]

# ── Industries & project templates ───────────────────────────────────────────
INDUSTRIES = {
    "E-commerce": {
        "hot_projects":  ["B2C e-commerce platform","multi-vendor marketplace","D2C brand website with mobile app","B2B wholesale portal"],
        "warm_projects": ["online store with basic features","product catalog website","e-commerce mobile app"],
        "cold_projects": ["simple online shop","small business website","product listing page"],
        "hot_services":  ["Web Development","Mobile App Development","Backend Development","UI/UX Design","API Integration"],
        "warm_services": ["Web Development","Mobile App Development","UI/UX Design","Backend Development"],
        "cold_services": ["Web Development","UI/UX Design"],
    },
    "FinTech": {
        "hot_projects":  ["digital banking portal with KYC","payment gateway integration platform","investment and wealth management app","insurance premium management system","NBFC loan origination system"],
        "warm_projects": ["payment collection app","invoicing and billing platform","expense tracking tool","basic accounting software"],
        "cold_projects": ["simple invoice generator","basic accounting tracker","monthly expense sheet app"],
        "hot_services":  ["Web Development","Backend Development","Cybersecurity Audit","API Integration","Cloud Infrastructure"],
        "warm_services": ["Web Development","Backend Development","API Integration"],
        "cold_services": ["Web Development"],
    },
    "Healthcare": {
        "hot_projects":  ["telemedicine platform with video consultation","hospital management system (HMS)","EMR / EHR system with HL7 compliance","pharmacy management and inventory system","doctor appointment booking with payments"],
        "warm_projects": ["clinic appointment booking app","patient records management","medical store inventory","health and wellness app"],
        "cold_projects": ["basic appointment reminder app","health tips website","doctor directory listing"],
        "hot_services":  ["Mobile App Development","Web Development","Backend Development","UI/UX Design","Database Design"],
        "warm_services": ["Mobile App Development","Web Development","Backend Development"],
        "cold_services": ["Web Development","Mobile App Development"],
    },
    "EdTech": {
        "hot_projects":  ["full LMS with live classes and assessments","online coaching platform with payment and analytics","skill development portal with certification","corporate training management system","K-12 learning app with parent dashboard"],
        "warm_projects": ["online tutoring booking platform","e-learning course website","quiz and assessment app","video lecture streaming platform"],
        "cold_projects": ["basic quiz app","simple educational website","study material sharing platform"],
        "hot_services":  ["Web Development","Mobile App Development","Backend Development","UI/UX Design","Cloud Infrastructure"],
        "warm_services": ["Web Development","Mobile App Development","Backend Development"],
        "cold_services": ["Web Development","Mobile App Development"],
    },
    "Logistics": {
        "hot_projects":  ["fleet management with real-time GPS tracking","warehouse management system (WMS)","last-mile delivery tracking platform","freight management and TMS","supply chain visibility dashboard"],
        "warm_projects": ["delivery tracking app","vehicle management tool","route optimization software","courier dispatch system"],
        "cold_projects": ["basic delivery status tracker","simple logistics website","driver task app"],
        "hot_services":  ["Mobile App Development","Backend Development","Web Development","API Integration","Cloud Infrastructure"],
        "warm_services": ["Mobile App Development","Backend Development","Web Development"],
        "cold_services": ["Mobile App Development","Web Development"],
    },
    "Food Tech": {
        "hot_projects":  ["cloud kitchen aggregator app like Swiggy for Tier-2 cities","restaurant management and POS system","multi-restaurant food ordering platform","chef-on-demand marketplace","corporate food ordering and delivery platform"],
        "warm_projects": ["restaurant ordering app","food delivery app for single brand","table booking platform","online menu and ordering system"],
        "cold_projects": ["simple food menu website","basic restaurant website","recipe sharing app"],
        "hot_services":  ["Mobile App Development","Backend Development","Web Development","UI/UX Design","API Integration"],
        "warm_services": ["Mobile App Development","Web Development","Backend Development"],
        "cold_services": ["Web Development","UI/UX Design"],
    },
    "Real Estate": {
        "hot_projects":  ["property listing and search portal with map integration","real estate CRM for agents and brokers","rental management platform","construction project management system","co-living space management platform"],
        "warm_projects": ["property listing website","real estate agent mobile app","lead management for brokers","rent collection app"],
        "cold_projects": ["basic property directory website","simple rental listing page","real estate blog"],
        "hot_services":  ["Web Development","Backend Development","Mobile App Development","UI/UX Design","API Integration"],
        "warm_services": ["Web Development","Backend Development","Mobile App Development"],
        "cold_services": ["Web Development"],
    },
    "Manufacturing": {
        "hot_projects":  ["enterprise ERP for multi-plant operations","IoT-based factory floor monitoring","supply chain management and vendor portal","quality management system (QMS)","production planning and scheduling tool"],
        "warm_projects": ["inventory management system","production tracking app","supplier management portal","basic ERP for small factory"],
        "cold_projects": ["simple inventory tracker","spreadsheet replacement app","basic order management"],
        "hot_services":  ["Web Development","Backend Development","Cloud Infrastructure","Database Design","API Integration"],
        "warm_services": ["Web Development","Backend Development","Database Design"],
        "cold_services": ["Web Development","Backend Development"],
    },
    "Travel & Tourism": {
        "hot_projects":  ["OTA travel booking platform with GDS integration","hotel property management system","tour package builder and booking engine","corporate travel management and expense platform","travel agency CRM with itinerary builder"],
        "warm_projects": ["travel booking website","hotel booking app","tour package listing site","trip planning app"],
        "cold_projects": ["travel blog website","basic tour listing page","destination guide app"],
        "hot_services":  ["Web Development","Mobile App Development","Backend Development","API Integration","UI/UX Design"],
        "warm_services": ["Web Development","Mobile App Development","Backend Development"],
        "cold_services": ["Web Development"],
    },
    "Retail": {
        "hot_projects":  ["omni-channel retail POS with inventory sync","loyalty and rewards management platform","retail analytics and BI dashboard","franchise management system","B2B wholesale ordering portal"],
        "warm_projects": ["retail POS system","inventory management app","loyalty program app","retail store website with online ordering"],
        "cold_projects": ["basic billing software","product catalog website","simple loyalty punch card app"],
        "hot_services":  ["Web Development","Backend Development","Mobile App Development","Database Design","API Integration"],
        "warm_services": ["Web Development","Backend Development","Mobile App Development"],
        "cold_services": ["Web Development","UI/UX Design"],
    },
    "AgriTech": {
        "hot_projects":  ["precision farming platform with IoT sensors","farmer advisory and market price app","agri supply chain and procurement platform","crop insurance claims management system","agri input distribution management"],
        "warm_projects": ["mandi price and weather app","farmer community app","crop disease detection app","agri e-commerce for inputs"],
        "cold_projects": ["basic weather app for farmers","crop calendar website","farming tips app"],
        "hot_services":  ["Mobile App Development","Backend Development","Web Development","API Integration","UI/UX Design"],
        "warm_services": ["Mobile App Development","Web Development","Backend Development"],
        "cold_services": ["Mobile App Development","Web Development"],
    },
    "SaaS / B2B Tools": {
        "hot_projects":  ["multi-tenant B2B SaaS CRM platform","HR and payroll management SaaS","project management and collaboration tool","B2B sales intelligence platform","subscription billing and revenue management SaaS"],
        "warm_projects": ["CRM for small teams","HR attendance and leave management app","task management tool","B2B lead generation tool"],
        "cold_projects": ["simple to-do app","basic team chat tool","note-taking app"],
        "hot_services":  ["Web Development","Backend Development","Cloud Infrastructure","UI/UX Design","Database Design"],
        "warm_services": ["Web Development","Backend Development","UI/UX Design"],
        "cold_services": ["Web Development"],
    },
}

# ── Designations ──────────────────────────────────────────────────────────────
HOT_DESIGNATIONS  = ["CEO","CTO","Co-Founder","Founder","Managing Director","VP Engineering","Head of Technology","Director - Digital","VP Product","Chief Digital Officer"]
WARM_DESIGNATIONS = ["Product Manager","Senior Manager","IT Manager","Marketing Manager","Operations Head","Business Analyst","Team Lead - Technology","Digital Manager","Assistant VP"]
COLD_DESIGNATIONS = ["Freelancer","Student","Self Employed","Junior Developer","Intern","Research Scholar","Assistant Manager","Executive"]

# ── Budget formats ────────────────────────────────────────────────────────────
def fmt_budget(lo_l: float, hi_l: float) -> str:
    """Format a budget range in INR lakhs as a string."""
    def to_str(x: float) -> str:
        paise = int(round(x * 100_000))
        # Format with Indian grouping
        s = str(paise)
        if len(s) <= 3:
            return s
        # Indian format: last 3 digits, then groups of 2
        result = s[-3:]
        s = s[:-3]
        while s:
            result = s[-2:] + "," + result
            s = s[:-2]
        return result.lstrip(",")
    return f"\u20b9{to_str(lo_l)},000 - \u20b9{to_str(hi_l)},000" if lo_l < 1 else f"\u20b9{to_str(lo_l)} - \u20b9{to_str(hi_l)}"

def rand_budget_hot() -> str:
    lo = random.choice([8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50])
    hi = lo + random.choice([4, 5, 8, 10, 15, 20])
    return f"\u20b9{lo},00,000 - \u20b9{hi},00,000"

def rand_budget_warm() -> str:
    lo = random.choice([2, 3, 4, 5, 6, 7])
    hi = lo + random.choice([2, 3, 4, 5])
    return f"\u20b9{lo},00,000 - \u20b9{hi},00,000"

def rand_budget_cold() -> str:
    options = [
        "", "",  # no budget given
        "\u20b950,000 - \u20b91,00,000",
        "\u20b930,000 - \u20b960,000",
        "\u20b910,000 - \u20b925,000",
        "\u20b95,000 - \u20b915,000",
        "\u20b91,00,000 - \u20b92,00,000",
        "Not decided",
        "Very limited budget",
    ]
    return random.choice(options)

# ── Company sizes ─────────────────────────────────────────────────────────────
def rand_size_hot():
    return random.choices(
        ["51-200","201-500","501-1000","1000+"],
        weights=[0.20, 0.35, 0.28, 0.17]
    )[0]

def rand_size_warm():
    return random.choices(
        ["11-50","51-200","201-500"],
        weights=[0.30, 0.50, 0.20]
    )[0]

def rand_size_cold():
    return random.choices(
        ["1-10","11-50","51-200"],
        weights=[0.65, 0.28, 0.07]
    )[0]

# ── Lead sources ──────────────────────────────────────────────────────────────
def rand_source_hot():
    return random.choices(
        ["Website Contact Form","Referral","Inbound Call","LinkedIn"],
        weights=[0.45, 0.38, 0.12, 0.05]
    )[0]

def rand_source_warm():
    return random.choices(
        ["Website Contact Form","LinkedIn","Google Search","Social Media Ad","Referral"],
        weights=[0.30, 0.25, 0.20, 0.15, 0.10]
    )[0]

def rand_source_cold():
    return random.choices(
        ["Google Search","Cold Outbound","Social Media Ad","Instagram Ad","Referral"],
        weights=[0.35, 0.30, 0.20, 0.10, 0.05]
    )[0]

# ── Query / message / description templates ───────────────────────────────────
HOT_QUERIES = [
    "We need to develop a {project} for our {company_type}. Budget has been approved and we are ready to start.",
    "We are looking for an experienced development team to build our {project}. We have detailed requirements and a clear timeline.",
    "Our company needs a {project}. We have secured funding and want to launch within the next month.",
    "We want to build a complete {project} solution. This is a high-priority initiative for us this quarter.",
    "We are urgently looking for a technology partner to develop our {project}. Board approval is done and budget is allocated.",
    "We need a reliable team to build our {project} from scratch. We have a clear specification and are ready to onboard immediately.",
    "Our {company_type} is planning to launch a {project} and we need a strong technical team. Timeline is firm.",
    "We are seeking a development partner for our {project}. This is a critical business project with approved funding.",
]

HOT_MESSAGES = [
    "We have evaluated a few vendors and are looking for the right fit. Our CTO has signed off on the budget. Can we schedule a technical discussion this week? We want to start the project within 2-3 weeks.",
    "This is a priority initiative for us. We already have wireframes and a product specification document ready. Looking to onboard a team ASAP. Budget is not a constraint for the right team.",
    "We have been planning this for 6 months and the investment committee has finally approved. We want to launch before the next quarter. Please share your team structure, development process and past work in this domain.",
    "We are a funded startup and need to ship our MVP fast. Our investors are watching the timeline closely. Please share your earliest available slot for a kickoff call.",
    "We need this built urgently. The product is the core of our business plan. We are flexible on budget for the right team. Previous vendor failed to deliver — we need someone serious and accountable.",
    "Budget is approved and the project is a top priority. We have completed the business analysis and user research. Looking for a team that can match our pace and quality standards.",
    "We have a runway of 18 months and a dedicated product budget. The project needs to launch in {timeline}. Please send your NDA and we can share the detailed requirements document.",
    "Our business depends on this platform. We cannot afford delays. We have an in-house product manager who will work closely with your team. Looking for a serious development partner.",
]

HOT_DESCRIPTIONS = [
    "The {project} needs to include: user authentication with OTP, {feature1}, {feature2}, payment gateway integration (Razorpay/PayU), an admin dashboard with analytics, and push notifications. The system must handle 10,000+ concurrent users and be built on a microservices architecture.",
    "Core modules required: (1) User onboarding with KYC verification, (2) {feature1} with real-time updates, (3) {feature2} with role-based access, (4) Analytics and reporting dashboard, (5) API integration with {api}. Must be scalable to 1 lakh users in 6 months.",
    "We need a complete {project} with: mobile apps (iOS + Android), a web admin panel, {feature1}, {feature2}, third-party API integrations ({api}), automated notifications, and a BI dashboard. Database design must support multi-tenancy.",
    "Technical requirements: React Native for mobile, Node.js/Django for backend, PostgreSQL for database, AWS for cloud hosting. Features: {feature1}, {feature2}, {feature3}, real-time data sync, offline mode for mobile app, and comprehensive audit logging.",
    "The platform must have: (a) Multi-role user management (admin, manager, field staff, customer), (b) {feature1}, (c) {feature2}, (d) Integration with {api}, (e) White-label capability, (f) SLA-grade uptime of 99.9%. We have a detailed BRD document to share.",
]

WARM_QUERIES = [
    "We are planning to build a {project} for our {company_type}. Could you share your approach and rough estimate?",
    "We need a {project} solution. We are currently evaluating vendors and looking for the right partner.",
    "Looking to develop a {project} for our business. We have some requirements but are open to suggestions.",
    "We want to create a {project}. Budget is partially allocated and we can discuss the rest.",
    "Our {company_type} is considering building a {project}. Would like to know more about your experience in this area.",
    "We are planning a {project} project. The timeline is flexible but we would like to get started in the next 2-3 months.",
    "We need a vendor to build our {project}. We are comparing a few options and would like to understand your process.",
    "We have been thinking about a {project} for a while. Could you share your past work and cost estimates?",
]

WARM_MESSAGES = [
    "We are in the evaluation phase and comparing 3-4 vendors. Budget is approved but not finalized for all features. Please share your portfolio with relevant case studies. We would like to make a decision within 2 weeks.",
    "We have a general idea of what we need but are open to your recommendations on the best approach. Budget is available but we want to ensure it fits the scope. Can you send your pricing structure?",
    "We want to move forward on this project this quarter. Please share your team size, relevant experience, and a rough cost estimate. We will follow up with a detailed requirement document once we select a vendor.",
    "Our management has shown interest in this project and we are gathering quotes. Please send your company profile, past work samples, and an estimated timeline. We expect to make a decision by end of this month.",
    "We have a budget allocated for this project. Looking for a team that has done similar work before. Can you share 2-3 relevant case studies and your development methodology?",
    "We are serious about moving forward but need to align internally on the exact features first. Please share your process for requirement gathering and how you manage projects. We'll schedule a call next week.",
    "The project is approved in principle. We are now shortlisting vendors. Your pricing, experience, and team structure will be key factors in our decision. Please respond with a capability deck.",
    "We would like to start the project in the next 2 months. Can you share your availability and a ballpark estimate so we can plan our budget accordingly?",
]

WARM_DESCRIPTIONS = [
    "We need a {project} with the following main features: user registration and login, {feature1}, {feature2}, and basic reporting. The exact scope will be refined during the requirement discussion phase.",
    "The {project} should have: a user-facing interface, {feature1}, and admin panel for management. Integration with existing systems may be required. We are open to your suggestions on the technology stack.",
    "Key requirements: {feature1}, {feature2}, and a mobile-friendly design. We also need training and post-launch support. The exact technical approach is open for discussion.",
    "We need the following features at minimum: {feature1}, user management, basic analytics. Additional features will depend on the budget. Please help us prioritize.",
    "The project requires a {project} that handles {feature1} and {feature2}. We do not have a technical team in-house so we need a vendor who can guide us through the process.",
]

COLD_QUERIES = [
    "I am looking for a {project}. What would be the approximate cost?",
    "Just exploring options for a {project}. No fixed timeline yet.",
    "Interested in building a {project}. What are your rates?",
    "Can you help with a {project}? We have a very small budget.",
    "I want to build something like a {project}. Is this possible within a limited budget?",
    "Looking for a developer to build a {project}. Price is very important for me.",
    "I need a simple {project}. How much would it cost?",
    "Curious about developing a {project}. We are still at the idea stage.",
]

COLD_MESSAGES = [
    "I am just exploring options right now. No firm decision has been made. Could you share a rough quote?",
    "Budget is very tight. We are a small team and cannot spend much. Looking for the most affordable option.",
    "We are at an early stage of planning. Not sure about the exact features yet. Will share more details once we have clarity.",
    "Just researching for now. No timeline or budget has been finalized. Wanted to understand the market rates.",
    "We are a small startup with very limited funds. Looking for the cheapest way to get this done. Can you offer a discount?",
    "This is a personal project. Not sure when we will actually start. Just wanted to get an idea of the cost.",
    "We might proceed with this in the future, not sure. Just gathering information for now.",
    "Budget is under 1 lakh. If not possible in this range, we will delay the project.",
]

COLD_DESCRIPTIONS = [
    "Just need a basic version of a {project}. Simple UI, minimal features. Will add more later if it works.",
    "We want something like {project} but very basic. No complex features needed for now.",
    "Not sure about the exact requirements yet. We will figure out the features once we get a quote.",
    "We want a {project} but have not thought about the technical details. Whatever is cheapest and fastest.",
    "Simple {project} with login, basic functionality. Nothing complex. For personal/small business use only.",
]

# ── Feature templates ─────────────────────────────────────────────────────────
FEATURES_BY_INDUSTRY = {
    "E-commerce":         ["product listing with filters","cart and checkout with Razorpay integration","order tracking with SMS notifications","multi-vendor seller panel","inventory management","wishlist and compare","product reviews and ratings"],
    "FinTech":            ["KYC with Aadhaar/PAN verification","real-time transaction ledger","loan origination with credit scoring","UPI and payment gateway integration","risk dashboard with alerts","multi-currency support","regulatory compliance reporting"],
    "Healthcare":         ["video consultation via WebRTC","e-prescription generation","OPD queue management","ABHA health ID integration","drug interaction checker","appointment reminder via WhatsApp","patient vitals tracking"],
    "EdTech":             ["live class via Zoom SDK integration","recorded video with HLS streaming","quiz builder with auto-grading","certificate generation","student progress tracking","parent dashboard","fee collection portal"],
    "Logistics":          ["real-time GPS tracking","POD with OTP and photo","route optimization","driver performance dashboard","automated dispatch","fuel consumption tracking","customer delivery notification"],
    "Food Tech":          ["restaurant onboarding with menu management","order flow with kitchen display system","dynamic pricing and promotions","delivery partner app","split payment and commission","real-time order tracking","review and rating system"],
    "Real Estate":        ["property listing with map and filters","virtual tour integration","EMI calculator","agent commission tracking","lead capture with follow-up CRM","document verification","rent agreement e-signing"],
    "Manufacturing":      ["BOM and routing management","shop floor execution","quality inspection checklists","vendor management portal","IoT sensor data ingestion","production yield tracking","maintenance request system"],
    "Travel & Tourism":   ["GDS/Amadeus flight search integration","hotel aggregator API","dynamic pricing for packages","booking and cancellation workflow","B2B agent portal with markup","travel CRM with itinerary builder","payment with partial booking"],
    "Retail":             ["POS with barcode scanner","multi-location inventory sync","loyalty points and rewards","GST-compliant billing","supplier order management","daily sales report","theft and shrinkage tracking"],
    "AgriTech":           ["live mandi price via API","7-day hyper-local weather forecast","push alerts for price threshold","chatbot for crop advisory in Hindi","input purchase marketplace","farm-to-fork traceability","government scheme discovery"],
    "SaaS / B2B Tools":   ["multi-tenant architecture with org isolation","SSO with Google/Microsoft","subscription billing via Stripe/Razorpay","role-based permission system","in-app notifications","API-first design with Swagger docs","audit trail and data export"],
}

APIS_BY_INDUSTRY = {
    "E-commerce":         "Razorpay, Shiprocket, Delhivery",
    "FinTech":            "RazorpayX, Setu, Perfios, Bureau.id",
    "Healthcare":         "Practo, ABHA, eSanjeevani",
    "EdTech":             "Zoom SDK, Vimeo, Juspay",
    "Logistics":          "Google Maps, Ola Maps, TruckMandi",
    "Food Tech":          "Razorpay, Pidge, Dunzo for Business",
    "Real Estate":        "99acres API, Google Maps, Leegality",
    "Manufacturing":      "Tally ERP, SAP B1, AWS IoT",
    "Travel & Tourism":   "Amadeus GDS, HotelBeds, Visa2fly",
    "Retail":             "Tally, GST Portal, Unicommerce",
    "AgriTech":           "IMD Weather, Agmarknet, APEDA",
    "SaaS / B2B Tools":   "Razorpay, Stripe, SendGrid, Twilio",
}

COMPANY_TYPES = {
    "E-commerce":         "retail business","FinTech":       "financial services company",
    "Healthcare":         "healthcare organization","EdTech": "education organization",
    "Logistics":          "logistics company","Food Tech":    "food business",
    "Real Estate":        "real estate company","Manufacturing":"manufacturing company",
    "Travel & Tourism":   "travel company","Retail":         "retail chain",
    "AgriTech":           "agri-business","SaaS / B2B Tools":"software company",
}

# ── Duration by tier ──────────────────────────────────────────────────────────
HOT_DURATIONS  = ["4-6 months","6-8 months","8-12 months","3-5 months","10-14 months"]
WARM_DURATIONS = ["3-4 months","2-3 months","4-5 months","5-7 months"]
COLD_DURATIONS = ["1-2 months","2-3 weeks","Unknown","3-4 weeks","Not decided"]

HOT_START_DATES  = ["Immediately","Within 1 month","Within 1 month","As soon as possible"]
WARM_START_DATES = ["Within 2 months","Next quarter","Within 3 months","Next 2-3 months"]
COLD_START_DATES = ["Anytime","No timeline","Next 6 months","Not decided","After the new year"]

HOT_TIMELINES  = ["3 months","4 months","end of quarter"]
WARM_TIMELINES = ["6 months","this year","next quarter"]

CONTACT_METHODS = ["Phone","Email","Video Call","WhatsApp"]


def _pick_features(industry: str, n: int) -> list:
    feats = FEATURES_BY_INDUSTRY.get(industry, ["core functionality","user management","reporting dashboard"])
    return random.sample(feats, min(n, len(feats)))


def _make_description(tier: str, project: str, industry: str) -> str:
    feats  = _pick_features(industry, 3)
    api    = APIS_BY_INDUSTRY.get(industry, "third-party APIs")
    tmpl   = random.choice({"Hot": HOT_DESCRIPTIONS, "Warm": WARM_DESCRIPTIONS, "Cold": COLD_DESCRIPTIONS}[tier])
    desc   = tmpl.format(
        project=project, feature1=feats[0],
        feature2=feats[1] if len(feats)>1 else "dashboard reporting",
        feature3=feats[2] if len(feats)>2 else "analytics",
        api=api
    )
    return desc


def _make_lead(tier: str, lead_id: str) -> dict:
    industry  = random.choice(list(INDUSTRIES.keys()))
    ind_data  = INDUSTRIES[industry]
    tier_key  = tier.lower()

    project   = random.choice(ind_data[f"{tier_key}_projects"])
    services  = ind_data[f"{tier_key}_services"]
    if tier == "Hot":
        num_svc = random.choices([3,4,5], weights=[0.25,0.45,0.30])[0]
    elif tier == "Warm":
        num_svc = random.choices([2,3],   weights=[0.55,0.45])[0]
    else:
        num_svc = random.choices([0,1],   weights=[0.25,0.75])[0]
    chosen_services = random.sample(services, min(num_svc, len(services)))

    # Name & contact
    fname = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)
    name  = f"{fname} {lname}"
    email = f"{fname.lower()}.{lname.lower()}@{random.choice(['gmail.com','company.in','business.co','corp.io','tech.com'])}"
    phone = f"+91-{random.randint(70,99)}{random.randint(10000000,99999999)}"

    # Company
    cname = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
    city  = random.choice(CITIES)

    # Tier-specific fields
    if tier == "Hot":
        size          = rand_size_hot()
        budget        = rand_budget_hot()
        urgency       = "High"
        dm            = random.random() < 0.90
        source        = rand_source_hot()
        designation   = random.choice(HOT_DESIGNATIONS)
        start_date    = random.choice(HOT_START_DATES)
        duration      = random.choice(HOT_DURATIONS)
        follow_up     = random.choices([0,1,2], weights=[0.55,0.30,0.15])[0]
        prev_contact  = follow_up > 0
        query_tmpl    = random.choice(HOT_QUERIES)
        msg_tmpl      = random.choice(HOT_MESSAGES)
        timeline      = random.choice(HOT_TIMELINES)

    elif tier == "Warm":
        size          = rand_size_warm()
        budget        = rand_budget_warm()
        urgency       = random.choices(["High","Medium","Medium"], weights=[0.25,0.50,0.25])[0]
        dm            = random.random() < 0.60
        source        = rand_source_warm()
        designation   = random.choice(WARM_DESIGNATIONS)
        start_date    = random.choice(WARM_START_DATES)
        duration      = random.choice(WARM_DURATIONS)
        follow_up     = random.choices([0,1], weights=[0.70,0.30])[0]
        prev_contact  = follow_up > 0
        query_tmpl    = random.choice(WARM_QUERIES)
        msg_tmpl      = random.choice(WARM_MESSAGES)
        timeline      = random.choice(WARM_TIMELINES)

    else:  # Cold
        size          = rand_size_cold()
        budget        = rand_budget_cold()
        urgency       = random.choices(["Low","Medium"], weights=[0.80,0.20])[0]
        dm            = random.random() < 0.70
        source        = rand_source_cold()
        designation   = random.choice(COLD_DESIGNATIONS)
        start_date    = random.choice(COLD_START_DATES)
        duration      = random.choice(COLD_DURATIONS)
        follow_up     = 0
        prev_contact  = False
        query_tmpl    = random.choice(COLD_QUERIES)
        msg_tmpl      = random.choice(COLD_MESSAGES)
        timeline      = "some time in the future"

    comp_type = COMPANY_TYPES.get(industry, "organization")
    query     = query_tmpl.format(project=project, company_type=comp_type, timeline=timeline)
    message   = msg_tmpl.format(timeline=timeline) if "{timeline}" in msg_tmpl else msg_tmpl
    desc      = _make_description(tier, project, industry)

    # Random created_at in past 90 days
    days_ago = random.randint(1, 90)
    created  = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "lead_id":              lead_id,
        "name":                 name,
        "phone":                phone,
        "email":                email,
        "company_name":         cname,
        "company_size":         size,
        "industry":             industry,
        "designation":          designation,
        "location":             f"{city}, India",
        "query":                query,
        "service_required":     chosen_services,
        "project_description":  desc,
        "estimated_budget":     budget,
        "expected_start_date":  start_date,
        "project_duration":     duration,
        "urgency":              urgency,
        "decision_maker":       dm,
        "lead_source":          source,
        "message":              message,
        "follow_up_count":      follow_up,
        "previously_contacted": prev_contact,
        "status":               "New",
        "label":                tier,
        "created_at":           created,
    }


def generate(output_path: str = "dataset.json") -> None:
    leads = []
    counters = {"Hot": 0, "Warm": 0, "Cold": 0}

    tiers = (
        ["Hot"]  * N_HOT  +
        ["Warm"] * N_WARM +
        ["Cold"] * N_COLD
    )
    random.shuffle(tiers)

    for i, tier in enumerate(tiers, start=1):
        counters[tier] += 1
        lead_id = f"LD-GEN-{i:04d}"
        leads.append(_make_lead(tier, lead_id))

    # Save
    out = Path(output_path)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(leads, fh, ensure_ascii=False, indent=2)

    total = len(leads)
    print(f"Generated {total} leads -> {out.resolve()}")
    print(f"  Hot  : {counters['Hot']:>4}  ({counters['Hot']/total*100:.1f}%)")
    print(f"  Warm : {counters['Warm']:>4}  ({counters['Warm']/total*100:.1f}%)")
    print(f"  Cold : {counters['Cold']:>4}  ({counters['Cold']/total*100:.1f}%)")
    print(f"  Total: {total}")


if __name__ == "__main__":
    generate()
