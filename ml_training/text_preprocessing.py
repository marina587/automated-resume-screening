"""
Text Preprocessing Module
Handles cleaning, normalization, tokenization, and stemming/lemmatization of resume text.
Preserves technical terms (C++, C#, .NET) and supports section-aware weighting.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import Dict, List, Optional, Tuple

# Optional spaCy support
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# Canonical tokens for technologies stripped by naive punctuation removal.
# NOTE: Abbreviations like 'ml', 'ai', 'k8s' are expanded in
# normalize_skill_aliases() (called first in clean_text) and should NOT
# appear here to avoid conflicting normalization paths.
TECH_TERM_REPLACEMENTS: List[Tuple[str, str]] = [
    (r'\bc\+\+\b', ' cpp '),
    (r'\bc#\b', ' csharp '),
    (r'\b\.net\b', ' dotnet '),
    (r'\breact\.js\b', ' reactjs '),
    (r'\bnode\.js\b', ' nodejs '),
    (r'\bvue\.js\b', ' vuejs '),
    (r'\bnext\.js\b', ' nextjs '),
    (r'\bexpress\.js\b', ' expressjs '),
    (r'\bangular\.js\b', ' angularjs '),
    (r'\bscikit-learn\b', ' scikitlearn '),
    (r'\bci/cd\b', ' cicd '),
    (r'\bui/ux\b', ' uiux '),
]

# Version patterns: Python 3.11, TensorFlow 2.x -> python311, tensorflow2
VERSION_PATTERNS: List[Tuple[str, str]] = [
    (r'\bpython\s*3\.(\d{1,2})\b', r' python3\1 '),
    (r'\bpython\s*(\d)\.(\d+)\b', r' python\1\2 '),
    (r'\btensorflow\s*2(?:\.x)?\b', ' tensorflow2 '),
    (r'\btensorflow\s*(\d+(?:\.\d+)?)\b', r' tensorflow\1 '),
    (r'\bpytorch\s*(\d+(?:\.\d+)?)\b', r' pytorch\1 '),
    (r'\bjava\s*(\d{1,2})\b', r' java\1 '),
    (r'\breact\s*(\d+(?:\.\d+)?)\b', r' react\1 '),
]

SECTION_HEADERS = [
    'experience', 'work experience', 'professional experience', 'employment',
    'education', 'academic background', 'qualifications',
    'skills', 'technical skills', 'core competencies', 'technologies',
    'projects', 'certifications', 'summary', 'objective', 'profile',
]

SECTION_WEIGHTS = {
    'experience': 2.0,
    'skills': 2.0,
    'projects': 1.5,
    'education': 1.0,
    'certifications': 1.0,
    'summary': 0.8,
    'objective': 0.8,
    'profile': 0.8,
    'other': 0.5,
}

# Skill synonyms: any alias match maps to canonical skill name
SKILL_SYNONYM_GROUPS: Dict[str, List[str]] = {
    'Python': [r'\bpython\b', r'\bpy\b'],
    'Java': [r'\bjava\b(?!\s*script)'],
    'JavaScript': [r'\bjavascript\b', r'\bjs\b'],
    'TypeScript': [r'\btypescript\b', r'\bts\b'],
    'C++': [r'\bc\+\+\b', r'\bcpp\b'],
    'C#': [r'\bc#\b', r'\bcsharp\b'],
    'Go': [r'\bgolang\b', r'\bgo\s+lang\b', r'\bgo\b(?!\s*(?:to|lang))'],
    'Ruby': [r'\bruby\b'],
    'PHP': [r'\bphp\b'],
    'Swift': [r'\bswift\b'],
    'Kotlin': [r'\bkotlin\b'],
    'Rust': [r'\brust\b'],
    'SQL': [r'\bsql\b'],
    'NoSQL': [r'\bnosql\b'],
    'MongoDB': [r'\bmongodb\b', r'\bmongo\b'],
    'PostgreSQL': [r'\bpostgresql\b', r'\bpostgres\b'],
    'MySQL': [r'\bmysql\b'],
    'Redis': [r'\bredis\b'],
    'Elasticsearch': [r'\belasticsearch\b', r'\belastic\s+search\b'],
    'AWS': [r'\baws\b', r'\bamazon web services\b'],
    'Azure': [r'\bazure\b', r'\bmicrosoft azure\b'],
    'GCP': [r'\bgcp\b', r'\bgoogle cloud\b', r'\bgoogle cloud platform\b'],
    'Docker': [r'\bdocker\b'],
    'Kubernetes': [r'\bkubernetes\b', r'\bk8s\b'],
    'Terraform': [r'\bterraform\b'],
    'Jenkins': [r'\bjenkins\b'],
    'Git': [r'\bgit\b', r'\bgithub\b', r'\bgitlab\b'],
    'Linux': [r'\blinux\b', r'\bunix\b'],
    'React': [r'\breact\b', r'\breactjs\b', r'\breact\.js\b'],
    'Angular': [r'\bangular\b'],
    'Vue': [r'\bvue\b', r'\bvuejs\b', r'\bvue\.js\b'],
    'Node.js': [r'\bnode\.?js\b', r'\bnodejs\b'],
    'Django': [r'\bdjango\b'],
    'Flask': [r'\bflask\b'],
    'Spring': [r'\bspring\s+boot\b', r'\bspring\b'],
    'TensorFlow': [r'\btensorflow\b', r'\btensorflow2\b'],
    'PyTorch': [r'\bpytorch\b'],
    'Keras': [r'\bkeras\b'],
    'Scikit-learn': [r'\bscikit-?learn\b', r'\bscikitlearn\b'],
    'Pandas': [r'\bpandas\b'],
    'NumPy': [r'\bnumpy\b'],
    'Matplotlib': [r'\bmatplotlib\b'],
    'Tableau': [r'\btableau\b'],
    'Power BI': [r'\bpower\s*bi\b'],
    'Machine Learning': [
        r'\bmachine learning\b', r'\bmachinelearning\b', r'\bml\b',
    ],
    'Deep Learning': [r'\bdeep learning\b', r'\bdl\b'],
    'NLP': [r'\bnlp\b', r'\bnatural language processing\b'],
    'Computer Vision': [r'\bcomputer vision\b', r'\bcv\b(?!\s*engineer)'],
    'Data Analysis': [r'\bdata analysis\b', r'\bdata analytics\b'],
    'Statistics': [r'\bstatistics\b', r'\bstatistical\b'],
    'Agile': [r'\bagile\b'],
    'Scrum': [r'\bscrum\b'],
    'REST API': [r'\brest\b', r'\brestful\b'],
    'GraphQL': [r'\bgraphql\b'],
    'Microservices': [r'\bmicroservices\b', r'\bmicro-service\b'],
    'CI/CD': [r'\bci/cd\b', r'\bcicd\b', r'\bcontinuous integration\b'],
    'HTML': [r'\bhtml\b', r'\bhtml5\b'],
    'CSS': [r'\bcss\b', r'\bcss3\b'],
    'Figma': [r'\bfigma\b'],
    'UI/UX': [r'\bui/ux\b', r'\buiux\b', r'\buser experience\b', r'\buser interface\b'],
    '.NET': [r'\b\.net\b', r'\bdotnet\b'],
}


def normalize_skill_aliases(text: str) -> str:
    """Expand common skill abbreviations so TF-IDF and matchers see full terms."""
    if not text:
        return ''
    normalized = text.lower()
    replacements = [
        (r'\bml\b', 'machine learning'),
        (r'\bai\b', 'artificial intelligence'),
        (r'\bk8s\b', 'kubernetes'),
        (r'\bjs\b', 'javascript'),
        (r'\bts\b', 'typescript'),
        (r'\bcpp\b', 'c++'),
        (r'\bcsharp\b', 'c#'),
        (r'\bpostgres\b', 'postgresql'),
        (r'\bmongo\b', 'mongodb'),
    ]
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return normalized


def protect_technical_terms(text: str) -> str:
    """Normalize tech tokens and versions before generic cleaning."""
    if not text:
        return ''
    text = text.lower()
    for pattern, replacement in TECH_TERM_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for pattern, replacement in VERSION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Split resume into sections (experience, education, skills, etc.).
    Returns dict mapping section name to content.
    """
    if not text:
        return {'other': ''}

    lines = text.split('\n')
    sections: Dict[str, List[str]] = {}
    current_section = 'other'
    sections[current_section] = []

    header_pattern = re.compile(
        r'^[\s#*]*(' + '|'.join(re.escape(h) for h in SECTION_HEADERS) + r')\s*:?\s*$',
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        header_match = header_pattern.match(stripped)
        if header_match:
            header = header_match.group(1).lower()
            if 'experience' in header or 'employment' in header:
                current_section = 'experience'
            elif 'education' in header or 'academic' in header or 'qualification' in header:
                current_section = 'education'
            elif 'skill' in header or 'competenc' in header or 'technolog' in header:
                current_section = 'skills'
            elif 'project' in header:
                current_section = 'projects'
            elif 'certif' in header:
                current_section = 'certifications'
            elif header in ('summary', 'objective', 'profile'):
                current_section = header
            else:
                current_section = 'other'
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(stripped)

    return {k: ' '.join(v) for k, v in sections.items() if v}


def build_section_weighted_text(text: str, sections: Optional[Dict[str, str]] = None) -> str:
    """
    Repeat section content proportional to importance for ML features.
    Experience and skills sections contribute more tokens.
    """
    sections = sections or parse_resume_sections(text)
    if not sections:
        return text

    weighted_parts = []
    for section_name, content in sections.items():
        weight = SECTION_WEIGHTS.get(section_name, SECTION_WEIGHTS['other'])
        repeats = max(1, int(round(weight)))
        weighted_parts.extend([content] * repeats)
    return ' '.join(weighted_parts)


class TextPreprocessor:
    """Handles text cleaning and normalization for resume processing."""

    def __init__(
        self,
        use_spacy: bool = True,
        use_lemmatization: bool = True,
        preserve_technical_terms: bool = True,
        section_aware: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.spacy_model = spacy_model
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_lemmatization = use_lemmatization
        self.preserve_technical_terms = preserve_technical_terms
        self.section_aware = section_aware

        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Install: python -m spacy download {spacy_model}"
                )
                self.use_spacy = False

    def clean_text(self, text: str) -> str:
        """
        Clean raw resume text while preserving technical vocabulary.
        """
        if not text or not isinstance(text, str):
            return ""

        text = normalize_skill_aliases(text)

        if self.preserve_technical_terms:
            text = protect_technical_terms(text)
        else:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)

        # Keep letters, digits (for python311), and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def tokenize_and_stem(self, text: str) -> List[str]:
        words = text.split()
        return [self.stemmer.stem(word) for word in words]

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        words = text.split()
        return [self.lemmatizer.lemmatize(word) for word in words]

    def preprocess_with_spacy(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return ' '.join(tokens)

    def preprocess(self, text: str, return_tokens: bool = False) -> str:
        """
        Full preprocessing pipeline with optional section weighting.

        When section-aware mode is enabled but no clear sections are detected
        (everything falls into 'other'), the text is used as-is without dilution
        to avoid reducing signal for short or non-standard resume inputs.
        """
        raw_for_sections = text
        if self.section_aware:
            sections = parse_resume_sections(raw_for_sections)
            # Only apply section weighting if we detected meaningful sections.
            # If everything is 'other', skip weighting to avoid diluting signal.
            if len(sections) == 1 and 'other' in sections:
                text = raw_for_sections
            else:
                text = build_section_weighted_text(raw_for_sections, sections)

        cleaned = self.clean_text(text)

        if self.use_spacy:
            result = self.preprocess_with_spacy(cleaned)
        else:
            no_stopwords = self.remove_stopwords(cleaned)
            if self.use_lemmatization:
                tokens = self.tokenize_and_lemmatize(no_stopwords)
            else:
                tokens = self.tokenize_and_stem(no_stopwords)
            result = ' '.join(tokens)

        if return_tokens:
            return result.split()
        return result

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(text) for text in texts]


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract technical skills using synonym-aware pattern groups.
    """
    text_lower = normalize_skill_aliases(text)
    found_skills: List[str] = []

    for skill, patterns in SKILL_SYNONYM_GROUPS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_skills.append(skill)
                break

    return found_skills


if __name__ == "__main__":
    sample_resume = """
    Jane Smith - Software Engineer
    Email: jane@email.com | Phone: 555-0100

    SKILLS
    C++, C#, .NET, React.js, Python 3.11, TensorFlow 2.x, ML, Kubernetes (k8s)

    EXPERIENCE
    Senior Engineer at TechCo (2019 - Present) — 7+ years of experience
    Built APIs with Node.js and PostgreSQL.

    EDUCATION
    MS Computer Science, State University
    """

    preprocessor = TextPreprocessor(use_spacy=False, use_lemmatization=True)
    print("Sections:", parse_resume_sections(sample_resume))
    print("Preprocessed:", preprocessor.preprocess(sample_resume))
    print("Skills:", extract_skills_from_text(sample_resume))
