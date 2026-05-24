"""
Text Preprocessing Module
Handles cleaning, normalization, tokenization, and stemming/lemmatization of resume text.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Optional

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


class TextPreprocessor:
    """Handles text cleaning and normalization for resume processing."""
    
    def __init__(self, use_spacy: bool = False, use_lemmatization: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_spacy: Whether to use spaCy for preprocessing (slower but more accurate)
            use_lemmatization: Whether to use lemmatization instead of stemming
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_lemmatization = use_lemmatization
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw resume text by removing unwanted characters and normalizing.
        
        Args:
            text: Raw resume text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (simple pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove English stopwords from text.
        
        Args:
            text: Cleaned text
            
        Returns:
            Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """
        Tokenize text and apply stemming.
        
        Args:
            text: Text to process
            
        Returns:
            List of stemmed tokens
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return stemmed_words
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and apply lemmatization.
        
        Args:
            text: Text to process
            
        Returns:
            List of lemmatized tokens
        """
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return lemmatized_words
    
    def preprocess_with_spacy(self, text: str) -> str:
        """
        Preprocess text using spaCy (includes lemmatization).
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        doc = self.nlp(text)
        # Get lemmatized tokens, excluding stopwords and punctuation
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and token.text.strip()]
        return ' '.join(tokens)
    
    def preprocess(self, text: str, return_tokens: bool = False) -> str:
        """
        Full preprocessing pipeline for resume text.
        
        Args:
            text: Raw resume text
            return_tokens: If True, return list of tokens; otherwise return processed string
            
        Returns:
            Preprocessed text or tokens
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        # Use spaCy if available
        if self.use_spacy:
            result = self.preprocess_with_spacy(cleaned)
        else:
            # Remove stopwords
            no_stopwords = self.remove_stopwords(cleaned)
            
            # Apply stemming or lemmatization
            if self.use_lemmatization:
                tokens = self.tokenize_and_lemmatize(no_stopwords)
            else:
                tokens = self.tokenize_and_stem(no_stopwords)
            
            result = ' '.join(tokens)
        
        if return_tokens:
            return result.split()
        return result
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract common technical skills from resume text using keyword matching.
    
    Args:
        text: Resume text
        
    Returns:
        List of extracted skills
    """
    # Common technical skills dictionary
    skill_patterns = {
        'Python': r'\bpython\b',
        'Java': r'\bjava\b',
        'JavaScript': r'\bjavascript\b|\bjs\b',
        'TypeScript': r'\btypescript\b',
        'C++': r'\bc\+\+\b',
        'C#': r'\bc#\b',
        'Go': r'\bgolang\b|\bgo\b',
        'Ruby': r'\bruby\b',
        'PHP': r'\bphp\b',
        'Swift': r'\bswift\b',
        'Kotlin': r'\bkotlin\b',
        'Rust': r'\brust\b',
        'SQL': r'\bsql\b',
        'NoSQL': r'\bnosql\b',
        'MongoDB': r'\bmongodb\b',
        'PostgreSQL': r'\bpostgresql\b',
        'MySQL': r'\bmysql\b',
        'Redis': r'\bredis\b',
        'Elasticsearch': r'\belasticsearch\b',
        'AWS': r'\baws\b|\bamazon web services\b',
        'Azure': r'\bazure\b',
        'GCP': r'\bgcp\b|\bgoogle cloud\b',
        'Docker': r'\bdocker\b',
        'Kubernetes': r'\bkubernetes\b|\bk8s\b',
        'Terraform': r'\bterraform\b',
        'Jenkins': r'\bjenkins\b',
        'Git': r'\bgit\b',
        'Linux': r'\blinux\b',
        'React': r'\breact\b',
        'Angular': r'\bangular\b',
        'Vue': r'\bvue\.?js?\b',
        'Node.js': r'\bnode\.?js?\b',
        'Django': r'\bdjango\b',
        'Flask': r'\bflask\b',
        'Spring': r'\bspring\b',
        'TensorFlow': r'\btensorflow\b',
        'PyTorch': r'\bpytorch\b',
        'Keras': r'\bkeras\b',
        'Scikit-learn': r'\bscikit-?learn\b',
        'Pandas': r'\bpandas\b',
        'NumPy': r'\bnumpy\b',
        'Matplotlib': r'\bmatplotlib\b',
        'Tableau': r'\btableau\b',
        'Power BI': r'\bpower bi\b',
        'Machine Learning': r'\bmachine learning\b',
        'Deep Learning': r'\bdeep learning\b',
        'NLP': r'\bnlp\b|\bnatural language processing\b',
        'Computer Vision': r'\bcomputer vision\b',
        'Data Analysis': r'\bdata analysis\b',
        'Statistics': r'\bstatistics\b',
        'Agile': r'\bagile\b',
        'Scrum': r'\bscrum\b',
        'REST API': r'\brest\b',
        'GraphQL': r'\bgraphql\b',
        'Microservices': r'\bmicroservices\b',
        'CI/CD': r'\bci/cd\b|\bcontinuous integration\b',
        'HTML': r'\bhtml\b',
        'CSS': r'\bcss\b',
        'Figma': r'\bfigma\b',
        'UI/UX': r'\bui/ux\b|\buser experience\b|\buser interface\b'
    }
    
    text_lower = text.lower()
    found_skills = []
    
    for skill, pattern in skill_patterns.items():
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    return found_skills


if __name__ == "__main__":
    # Example usage
    sample_resume = """
    John Doe - Software Engineer
    Email: john.doe@email.com
    Phone: 123-456-7890
    
    Experienced software engineer with 5+ years of experience in Python, Java, and JavaScript.
    Proficient in machine learning, TensorFlow, and data analysis.
    Worked with AWS, Docker, Kubernetes for cloud deployment.
    Skilled in React, Node.js, and building REST APIs.
    Strong background in agile methodologies and CI/CD pipelines.
    https://github.com/johndoe
    """
    
    preprocessor = TextPreprocessor(use_spacy=False, use_lemmatization=True)
    
    print("Original text:")
    print(sample_resume)
    print("\n" + "="*50 + "\n")
    
    cleaned = preprocessor.clean_text(sample_resume)
    print("Cleaned text:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    preprocessed = preprocessor.preprocess(sample_resume)
    print("Preprocessed text:")
    print(preprocessed)
    print("\n" + "="*50 + "\n")
    
    skills = extract_skills_from_text(sample_resume)
    print(f"Extracted skills: {skills}")
