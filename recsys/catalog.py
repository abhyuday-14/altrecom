"""Static domain knowledge: topics, career goals and the 48-course catalogue.

Each course: (id, name, topic_index, level(year 1-4), difficulty 1-5, prerequisites, description)
"""

TOPICS = ["AI/ML", "Data", "Systems", "Web/Software", "Security", "Theory"]
TOPIC_IDX = {t: i for i, t in enumerate(TOPICS)}

# How strongly each career goal relates to every topic (rows follow TOPICS order).
CAREERS = {
    "Data Scientist":          [0.30, 0.45, 0.00, 0.05, 0.00, 0.20],
    "ML Engineer":             [0.55, 0.20, 0.10, 0.10, 0.00, 0.05],
    "Web/App Developer":       [0.00, 0.10, 0.05, 0.75, 0.10, 0.00],
    "Security Analyst":        [0.00, 0.00, 0.25, 0.10, 0.60, 0.05],
    "Systems Engineer":        [0.00, 0.05, 0.65, 0.15, 0.10, 0.05],
    "Research / Higher Studies": [0.30, 0.05, 0.10, 0.00, 0.05, 0.50],
}

# Dirichlet prior tilt per branch (cosmetic realism + gives CF something to learn).
BRANCHES = {
    "CSE":     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "IT":      [0.8, 1.0, 0.8, 1.6, 1.0, 0.6],
    "ECE":     [0.6, 0.6, 1.8, 0.8, 1.0, 0.8],
    "AI&DS":   [1.8, 1.6, 0.6, 0.8, 0.5, 0.8],
}

COURSES = [
    # ---- Year 1 ----
    ("CS101", "Programming with Python", 3, 1, 2, [], "python programming variables loops functions scripting problem solving"),
    ("CS102", "Discrete Mathematics", 5, 1, 3, [], "logic sets relations graphs combinatorics proofs"),
    ("CS103", "Linear Algebra for Computing", 5, 1, 3, [], "vectors matrices eigenvalues linear transformations applications"),
    ("CS104", "Probability and Statistics", 1, 1, 3, [], "probability random variables distributions hypothesis testing inference"),
    ("CS105", "Digital Logic Design", 2, 1, 2, [], "boolean algebra logic gates circuits flip flops"),
    ("CS106", "Web Fundamentals", 3, 1, 2, [], "html css javascript responsive web pages"),
    ("CS107", "Introduction to Data Analysis", 1, 1, 2, [], "spreadsheets data cleaning visualization charts exploratory analysis"),
    ("CS108", "Cyber Hygiene and Ethics", 4, 1, 1, [], "passwords privacy phishing safe computing ethics"),
    # ---- Year 2 ----
    ("CS201", "Data Structures", 3, 2, 3, ["CS101"], "arrays linked lists trees graphs hashing algorithms complexity"),
    ("CS202", "Database Management Systems", 1, 2, 3, ["CS101"], "sql relational model normalization transactions indexing"),
    ("CS203", "Computer Organization", 2, 2, 3, ["CS105"], "processor architecture instruction set memory hierarchy pipelining"),
    ("CS204", "Object Oriented Programming", 3, 2, 2, ["CS101"], "classes inheritance polymorphism design patterns java"),
    ("CS205", "Design and Analysis of Algorithms", 5, 2, 4, ["CS201", "CS102"], "divide conquer dynamic programming greedy graph algorithms np completeness"),
    ("CS206", "Python for Data Science", 1, 2, 2, ["CS101", "CS107"], "numpy pandas matplotlib data wrangling notebooks"),
    ("CS207", "Introduction to Artificial Intelligence", 0, 2, 3, ["CS101"], "search heuristics knowledge representation agents constraint satisfaction"),
    ("CS208", "Computer Networks", 2, 2, 3, ["CS105"], "tcp ip routing protocols layers sockets"),
    ("CS209", "Web Development with JavaScript", 3, 2, 2, ["CS106"], "javascript dom react frontend apis"),
    ("CS210", "Introduction to Cryptography", 4, 2, 3, ["CS102"], "encryption symmetric public key hashing signatures number theory"),
    ("CS211", "Statistical Methods for Data Analysis", 1, 2, 3, ["CS104"], "regression anova sampling confidence intervals statistical modelling"),
    ("CS212", "Theory of Computation", 5, 2, 4, ["CS102"], "automata regular languages turing machines decidability"),
    # ---- Year 3 ----
    ("CS301", "Machine Learning", 0, 3, 4, ["CS103", "CS104", "CS206"], "supervised unsupervised learning regression classification clustering model evaluation"),
    ("CS302", "Deep Learning", 0, 3, 5, ["CS301"], "neural networks backpropagation cnn rnn optimization pytorch"),
    ("CS303", "Natural Language Processing", 0, 3, 4, ["CS301"], "text tokenization embeddings language models sentiment translation"),
    ("CS304", "Computer Vision", 0, 3, 4, ["CS301"], "image processing convolution object detection segmentation"),
    ("CS305", "Operating Systems", 2, 3, 4, ["CS201", "CS203"], "processes threads scheduling memory management file systems deadlocks"),
    ("CS306", "Data Mining and Warehousing", 1, 3, 3, ["CS202", "CS206"], "association rules clustering classification olap data warehouse etl"),
    ("CS307", "Big Data Analytics", 1, 3, 4, ["CS202", "CS206"], "hadoop spark mapreduce distributed processing streaming"),
    ("CS308", "Data Visualization", 1, 3, 2, ["CS206"], "dashboards storytelling tableau plotting visual analytics"),
    ("CS309", "Software Engineering", 3, 3, 2, ["CS204"], "requirements agile testing uml version control project management"),
    ("CS310", "Full Stack Web Development", 3, 3, 3, ["CS209", "CS202"], "node express mongodb rest api authentication deployment"),
    ("CS311", "Mobile App Development", 3, 3, 3, ["CS204"], "android flutter user interface sensors app store"),
    ("CS312", "Network Security", 4, 3, 4, ["CS208", "CS210"], "firewalls intrusion detection vpn ssl attacks defense"),
    ("CS313", "Ethical Hacking and Penetration Testing", 4, 3, 4, ["CS208"], "vulnerability scanning exploits metasploit reconnaissance web attacks"),
    ("CS314", "Distributed Systems", 2, 3, 5, ["CS208", "CS201"], "replication consensus fault tolerance clocks microservices"),
    ("CS315", "Compiler Design", 5, 3, 5, ["CS212", "CS201"], "lexical analysis parsing code generation optimization"),
    ("CS316", "Cloud Computing", 2, 3, 3, ["CS208"], "virtualization aws containers docker serverless scalability"),
    # ---- Year 4 ----
    ("CS401", "Reinforcement Learning", 0, 4, 5, ["CS301"], "agents rewards markov decision processes q learning policy gradient"),
    ("CS402", "Generative AI and LLMs", 0, 4, 5, ["CS302"], "transformers large language models prompting diffusion fine tuning"),
    ("CS403", "Recommender Systems", 0, 4, 4, ["CS301"], "collaborative filtering content based matrix factorization ranking evaluation"),
    ("CS404", "MLOps and Model Deployment", 0, 4, 3, ["CS301"], "model deployment monitoring pipelines docker versioning"),
    ("CS405", "Time Series Forecasting", 1, 4, 4, ["CS211"], "arima seasonality forecasting trends prediction"),
    ("CS406", "Information Retrieval", 1, 4, 3, ["CS202"], "indexing search engines ranking relevance query"),
    ("CS407", "Cyber Forensics", 4, 4, 3, ["CS208", "CS108"], "digital evidence disk analysis incident response logs"),
    ("CS408", "Blockchain Technology", 4, 4, 3, ["CS210"], "distributed ledger consensus smart contracts ethereum"),
    ("CS409", "High Performance Computing", 2, 4, 5, ["CS305"], "parallel programming gpu cuda mpi performance"),
    ("CS410", "Embedded and IoT Systems", 2, 4, 3, ["CS203"], "microcontrollers sensors arduino iot protocols real time"),
    ("CS411", "Advanced Algorithms", 5, 4, 5, ["CS205"], "randomized approximation algorithms network flow amortized analysis"),
    ("CS412", "Quantum Computing", 5, 4, 5, ["CS103", "CS212"], "qubits quantum gates superposition algorithms shor grover"),
]
