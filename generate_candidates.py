"""
Script to generate 100 synthetic candidate profiles.
Run once: python generate_candidates.py
Output: data/candidates.json
"""
import json, random

first_names = [
    "Priya","Rahul","Sneha","Arjun","Divya","Karan","Ananya","Vikram","Meera","Rohan",
    "Pooja","Aditya","Nisha","Siddharth","Kavya","Amit","Shruti","Nikhil","Riya","Harsh",
    "Ankita","Varun","Swati","Deepak","Simran","Rajesh","Preeti","Suresh","Tanvi","Mohit",
    "Isha","Gaurav","Neha","Kunal","Payal","Abhishek","Ritika","Manish","Shweta","Vishal"
]
last_names = [
    "Sharma","Patel","Gupta","Singh","Kumar","Mehta","Joshi","Verma","Nair","Iyer",
    "Reddy","Rao","Shah","Malhotra","Bose","Ghosh","Pillai","Menon","Choudhury","Das"
]
locations = ["Bangalore","Mumbai","Pune","Hyderabad","Chennai","Delhi","Noida","Gurgaon","Kolkata","Ahmedabad"]

PROFILES = [
    {
        "track": "backend",
        "titles": ["Backend Engineer","Senior Backend Developer","Software Engineer – Backend","Backend Lead"],
        "skills_pool": ["Python","FastAPI","Django","Node.js","PostgreSQL","MySQL","Redis","Docker","AWS","GCP","Kubernetes","REST APIs","GraphQL","Celery","RabbitMQ"],
        "summaries": [
            "Experienced backend engineer specializing in high-throughput APIs and distributed systems. Passionate about clean architecture.",
            "Backend developer with deep expertise in Python and cloud infrastructure. Built services handling millions of requests daily.",
            "Full-stack leaning backend engineer who loves designing scalable microservices and mentoring junior developers."
        ]
    },
    {
        "track": "ml",
        "titles": ["ML Engineer","Senior Data Scientist","AI/ML Developer","Machine Learning Engineer"],
        "skills_pool": ["Python","TensorFlow","PyTorch","scikit-learn","NLP","LLMs","Transformers","HuggingFace","SQL","Spark","MLflow","Pandas","NumPy","OpenCV","LangChain"],
        "summaries": [
            "ML engineer focused on NLP and LLM fine-tuning. Shipped production recommendation systems at e-commerce scale.",
            "Data scientist with a track record of taking models from notebook to production. Expert in feature engineering and model monitoring.",
            "AI engineer specializing in generative AI applications. Built RAG pipelines and agent frameworks for enterprise clients."
        ]
    },
    {
        "track": "frontend",
        "titles": ["Frontend Engineer","Senior React Developer","UI Engineer","Frontend Lead"],
        "skills_pool": ["React","TypeScript","Next.js","Tailwind CSS","GraphQL","Redux","Webpack","Vite","Jest","Cypress","Figma","CSS3","HTML5","Vue.js","Storybook"],
        "summaries": [
            "Frontend engineer with a strong eye for design and performance optimization. Delivered pixel-perfect UIs for SaaS products.",
            "React specialist who bridges the gap between design and engineering. Deep experience with accessibility and animations.",
            "UI engineer passionate about developer experience and component library architecture. Built design systems from scratch."
        ]
    },
    {
        "track": "fullstack",
        "titles": ["Full Stack Developer","Senior Software Engineer","Software Engineer","Full Stack Engineer"],
        "skills_pool": ["React","Node.js","Python","TypeScript","PostgreSQL","MongoDB","Docker","AWS","REST APIs","GraphQL","Redis","Next.js","Express","Prisma","Terraform"],
        "summaries": [
            "Full stack engineer comfortable across the entire web stack. Loves shipping products end-to-end from DB schema to pixel.",
            "Generalist engineer with startup experience — moved fast, wore many hats, shipped things that matter.",
            "Senior engineer with equal comfort in frontend and backend. Led greenfield projects from architecture to deployment."
        ]
    },
    {
        "track": "devops",
        "titles": ["DevOps Engineer","Site Reliability Engineer","Platform Engineer","Cloud Engineer"],
        "skills_pool": ["Kubernetes","Docker","Terraform","AWS","GCP","Azure","CI/CD","Jenkins","GitHub Actions","Prometheus","Grafana","Ansible","Helm","Linux","Python"],
        "summaries": [
            "DevOps engineer obsessed with reliability and developer velocity. Reduced deployment times by 60% at last company.",
            "SRE with deep Kubernetes expertise. Designed multi-region failover systems for fintech platforms.",
            "Cloud platform engineer who built internal developer platforms that let product teams ship with confidence."
        ]
    },
    {
        "track": "product",
        "titles": ["Product Manager","Senior Product Manager","Technical Product Manager","Associate PM"],
        "skills_pool": ["Product Strategy","Roadmapping","User Research","SQL","Figma","Jira","A/B Testing","Data Analysis","Stakeholder Management","Agile","OKRs","Growth","Pricing","Go-to-market"],
        "summaries": [
            "PM with a technical background and a bias for shipping. Managed 0→1 products and scaled features to millions of users.",
            "Customer-obsessed product manager who combines qualitative research with quantitative analysis to drive decisions.",
            "Technical PM who can read code and sit with engineers. Shipped developer tools and APIs for B2B SaaS companies."
        ]
    },
    {
        "track": "agentic_ai",
        "titles": ["AI Engineer","LLM Application Developer","Agentic AI Developer","GenAI Engineer"],
        "skills_pool": ["LangChain","LlamaIndex","OpenAI API","Claude API","RAG","Vector Databases","ChromaDB","Pinecone","Python","FastAPI","Prompt Engineering","Tool Use","Multi-agent Systems","AutoGen","CrewAI"],
        "summaries": [
            "AI engineer specializing in agentic systems and RAG pipelines. Built production AI assistants for legal and HR domains.",
            "GenAI developer with hands-on experience building multi-agent workflows using LangChain and LlamaIndex.",
            "LLM engineer focused on tool-use, memory, and planning patterns in agent architectures."
        ]
    }
]

availabilities = ["Immediate","1 week","2 weeks","1 month","3 months"]
notice_weights = [0.1, 0.15, 0.35, 0.3, 0.1]

candidates = []
cid = 1

for _ in range(100):
    profile = random.choice(PROFILES)
    exp = random.randint(1, 12)
    
    # Pick skills - ensure 4-10 skills
    n_skills = random.randint(4, 10)
    skills = random.sample(profile["skills_pool"], min(n_skills, len(profile["skills_pool"])))
    
    # Seniority based on exp
    if exp <= 2:
        title = profile["titles"][-1] if len(profile["titles"]) > 2 else profile["titles"][0]
    elif exp <= 5:
        title = profile["titles"][0]
    else:
        title = profile["titles"][1] if len(profile["titles"]) > 1 else profile["titles"][0]

    candidate = {
        "id": f"c{cid:03d}",
        "name": f"{random.choice(first_names)} {random.choice(last_names)}",
        "title": title,
        "track": profile["track"],
        "skills": skills,
        "experience_years": exp,
        "location": random.choice(locations),
        "availability": random.choices(availabilities, notice_weights)[0],
        "expected_ctc_lpa": round(exp * random.uniform(1.2, 2.0), 1),
        "summary": random.choice(profile["summaries"]),
        "linkedin": f"linkedin.com/in/{random.choice(first_names).lower()}{random.randint(10,99)}",
        "open_to_remote": random.choice([True, True, False]),
        "open_to_relocation": random.choice([True, False])
    }
    candidates.append(candidate)
    cid += 1

with open("data/candidates.json", "w") as f:
    json.dump(candidates, f, indent=2)

print(f"Generated {len(candidates)} candidates → data/candidates.json")
