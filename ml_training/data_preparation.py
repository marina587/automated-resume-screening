"""
Data Collection and Preparation Module
Handles loading, exploring, and preprocessing the resume dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


# Category sampling weights (majority classes oversampled in raw data — balance later)
CATEGORY_WEIGHTS = {
    'Software Engineer': 0.22,
    'Backend Developer': 0.14,
    'Frontend Developer': 0.12,
    'Full Stack Developer': 0.10,
    'Data Scientist': 0.10,
    'Machine Learning Engineer': 0.08,
    'DevOps Engineer': 0.08,
    'Data Analyst': 0.07,
    'Product Manager': 0.05,
    'UX Designer': 0.04,
}

ROLE_TEMPLATES = {
    'Data Scientist': [
        "Data Scientist with {years}+ years applying Python {pyver}, scikit-learn, and {ml_stack}. "
        "Built predictive models using {cloud} and SQL. MS in Statistics. "
        "Skills: Pandas, NumPy, TensorFlow {tfver}, NLP, data visualization.",
        "Senior Data Scientist — {years} years in ML pipelines, A/B testing, and stakeholder reporting. "
        "Proficient in Python 3.{py_minor}, PyTorch, Spark, and Tableau. PhD coursework in machine learning.",
    ],
    'Software Engineer': [
        "Software Engineer ({years}+ yrs) — Java {javaver}, Spring Boot, microservices on {cloud}. "
        "CI/CD with Jenkins, Docker, Kubernetes. BS Computer Science.",
        "Backend-focused Software Engineer using C++, Python, and REST APIs. "
        "Experience with .NET Core, PostgreSQL, Redis, and Agile/Scrum teams since {start_year}.",
    ],
    'Product Manager': [
        "Product Manager with {years} years driving roadmaps, user research, and KPI tracking. "
        "Agile ceremonies, Jira, and cross-functional leadership with engineering and design.",
    ],
    'UX Designer': [
        "UX Designer — {years} years wireframing in Figma, usability testing, and design systems. "
        "Portfolio includes mobile and web UI/UX for SaaS products.",
    ],
    'DevOps Engineer': [
        "DevOps Engineer — {years}+ years: Terraform, Kubernetes, AWS, CI/CD, monitoring. "
        "Automated deployments with Docker and Python scripting.",
    ],
    'Data Analyst': [
        "Data Analyst ({years} yrs) — SQL, Power BI, Excel, statistical reporting for business units. "
        "Experience with Python for automation and dashboarding.",
    ],
    'Machine Learning Engineer': [
        "ML Engineer deploying models at scale — PyTorch, TensorFlow 2.x, MLOps on {cloud}. "
        "{years}+ years in computer vision and NLP production systems.",
    ],
    'Frontend Developer': [
        "Frontend Developer specializing in React.js, TypeScript, CSS3, and accessibility. "
        "{years} years building responsive SPAs with Node.js backends.",
    ],
    'Backend Developer': [
        "Backend Developer — Node.js, GraphQL, PostgreSQL, Redis. "
        "{years}+ years designing scalable APIs and event-driven architectures.",
    ],
    'Full Stack Developer': [
        "Full Stack Developer — MERN stack, GraphQL, MongoDB. "
        "{years} years end-to-end delivery with React and Express.js on {cloud}.",
    ],
}

HOLDOUT_STYLE_PREFIX = [
    "Professional summary: ",
    "Career profile — ",
    "Candidate overview | ",
]


class DataPreparator:
    """Handles data loading, exploration, and initial cleaning."""

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        path = file_path or self.data_path
        if not path:
            raise ValueError("No data path provided")

        self.df = pd.read_csv(path)
        print(f"Loaded {len(self.df)} resumes")
        return self.df

    def explore_data(self) -> dict:
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        imbalance_ratio = None
        if 'category' in self.df.columns:
            counts = self.df['category'].value_counts()
            if len(counts) > 1:
                imbalance_ratio = float(counts.max() / counts.min())

        results = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': int(self.df.duplicated().sum()),
            'class_distribution': (
                self.df['category'].value_counts().to_dict()
                if 'category' in self.df.columns else {}
            ),
            'imbalance_ratio': imbalance_ratio,
            'sample_data': self.df.head(3).to_dict(),
        }

        print(f"Dataset shape: {results['shape']}")
        print(f"Missing values: {results['missing_values']}")
        print(f"Duplicates: {results['duplicates']}")
        if imbalance_ratio:
            print(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
        print(
            f"Class distribution:\n"
            f"{self.df['category'].value_counts() if 'category' in self.df.columns else 'No category column'}"
        )
        return results

    def clean_data(self, category_column: str = None, text_column: str = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded")

        if text_column is None:
            possible_text_cols = [
                'resume_text', 'resume', 'text', 'content', 'description', 'resume_content',
            ]
            for col in possible_text_cols:
                if col in self.df.columns:
                    text_column = col
                    break
            if text_column is None:
                raise ValueError(
                    f"Could not find resume text column. Available: {list(self.df.columns)}"
                )

        if category_column is None:
            possible_cat_cols = [
                'category', 'label', 'class', 'job_title', 'job_category',
                'role', 'designation', 'job_role', 'position',
            ]
            for col in possible_cat_cols:
                if col in self.df.columns:
                    category_column = col
                    break
            if category_column is None:
                raise ValueError(
                    f"Could not find category column. Available: {list(self.df.columns)}"
                )

        print(f"Using text column: '{text_column}', category column: '{category_column}'")

        rename_map = {}
        if text_column != 'resume_text':
            rename_map[text_column] = 'resume_text'
        if category_column != 'category':
            rename_map[category_column] = 'category'
        if rename_map:
            self.df = self.df.rename(columns=rename_map)
            print(f"Renamed columns: {rename_map}")

        self.df['resume_text'] = self.df['resume_text'].astype(str)
        self.df['category'] = self.df['category'].astype(str)

        self.df = self.df.dropna(subset=['resume_text', 'category'])
        self.df = self.df[self.df['resume_text'].str.strip() != '']
        self.df = self.df[self.df['category'].str.strip() != '']
        self.df = self.df.drop_duplicates(subset=['resume_text'])

        print(f"Cleaned dataset: {len(self.df)} resumes remaining")
        return self.df

    def balance_classes(
        self,
        target_column: str = 'category',
        strategy: str = 'oversample',
        random_state: int = 42,
        only_train: bool = True,
        unknown_category: str = UNKNOWN_CATEGORY,
    ) -> pd.DataFrame:
        """
        Address class imbalance via oversampling minority classes or undersampling majority.
        When data_source exists, only rebalances 'train' rows so holdout stays untouched.

        The 'Unknown' category is excluded from oversampling/undersampling — it is kept
        at its original count to avoid diluting known-category signals.
        """
        if self.df is None:
            raise ValueError("No data loaded")

        rng = np.random.default_rng(random_state)

        def _balance_frame(frame: pd.DataFrame) -> pd.DataFrame:
            # Separate Unknown from known categories
            unknown_mask = frame[target_column] == unknown_category
            unknown_df = frame[unknown_mask]
            known_df = frame[~unknown_mask]

            if len(known_df) == 0:
                return frame

            groups = [group for _, group in known_df.groupby(target_column)]
            counts = [len(g) for g in groups]
            target_n = min(counts) if strategy == 'undersample' else max(counts)

            balanced_parts = []
            for group in groups:
                n = len(group)
                if n < target_n:
                    extra_idx = rng.choice(group.index, size=target_n - n, replace=True)
                    balanced_parts.append(pd.concat([group, frame.loc[extra_idx]]))
                elif n > target_n and strategy == 'undersample':
                    balanced_parts.append(
                        group.sample(n=target_n, random_state=random_state)
                    )
                else:
                    balanced_parts.append(group)

            # Append Unknown rows unchanged (not oversampled/undersampled)
            balanced_parts.append(unknown_df)
            return pd.concat(balanced_parts, ignore_index=True)

        if only_train and 'data_source' in self.df.columns:
            train_df = self.df[self.df['data_source'] == 'train']
            holdout_df = self.df[self.df['data_source'] == 'holdout']
            train_balanced = _balance_frame(train_df)
            self.df = pd.concat([train_balanced, holdout_df], ignore_index=True)
        else:
            self.df = _balance_frame(self.df)

        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"Balanced dataset ({strategy}): {len(self.df)} rows")
        print(self.df[target_column].value_counts())
        return self.df

    def assign_data_source(
        self,
        holdout_fraction: float = 0.2,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Tag rows as 'train' or 'holdout' for out-of-distribution evaluation.
        Holdout rows use a different text style when generated via create_sample_dataset.
        """
        if self.df is None:
            raise ValueError("No data loaded")

        if 'data_source' in self.df.columns:
            return self.df

        rng = np.random.default_rng(random_state)
        n_holdout = max(1, int(len(self.df) * holdout_fraction))
        holdout_idx = set(rng.choice(self.df.index, size=n_holdout, replace=False))

        self.df['data_source'] = [
            'holdout' if idx in holdout_idx else 'train'
            for idx in self.df.index
        ]
        print(
            f"Assigned data_source: "
            f"{(self.df['data_source'] == 'holdout').sum()} holdout, "
            f"{(self.df['data_source'] == 'train').sum()} train"
        )
        return self.df

    def save_cleaned_data(self, output_path: str = 'data/resumes_cleaned.csv'):
        if self.df is None:
            raise ValueError("No data to save")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        return output_path


def _render_template(category: str, template: str, rng: np.random.Generator, holdout: bool) -> str:
    years = int(rng.integers(2, 15))
    py_minor = int(rng.integers(8, 12))
    start_year = 2026 - years
    text = template.format(
        years=years,
        pyver=f"3.{py_minor}",
        py_minor=py_minor,
        tfver="2.x",
        javaver=int(rng.integers(8, 22)),
        ml_stack=rng.choice(["TensorFlow", "PyTorch", "XGBoost"]),
        cloud=rng.choice(["AWS", "Azure", "GCP"]),
        start_year=start_year,
    )
    extras = rng.choice([
        f" Certifications: {rng.choice(['AWS Certified', 'PMP', 'CKA'])}.",
        f" Tools: {rng.choice(['Jira', 'Confluence', 'Datadog'])}.",
        f" Languages: {rng.choice(['English (native)', 'Spanish (professional)'])}.",
        "",
    ])
    text = text + extras
    if holdout:
        text = rng.choice(HOLDOUT_STYLE_PREFIX) + text
    return text


def create_sample_dataset(
    output_path: str = 'data/sample_resumes.csv',
    n_samples: int = 1000,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
) -> str:
    """
    Create a varied sample dataset with realistic imbalance and holdout split markers.
    """
    rng = np.random.default_rng(random_state)
    categories = list(CATEGORY_WEIGHTS.keys())
    weights = [CATEGORY_WEIGHTS[c] for c in categories]

    n_holdout = max(1, int(n_samples * holdout_fraction))
    holdout_indices = set(rng.choice(n_samples, size=n_holdout, replace=False))

    data = []
    for i in range(n_samples):
        category = rng.choice(categories, p=weights)
        templates = ROLE_TEMPLATES[category]
        template = templates[int(rng.integers(0, len(templates)))]
        is_holdout = i in holdout_indices
        resume_text = _render_template(category, template, rng, holdout=is_holdout)

        # Add noise and unique tokens
        noise_skills = rng.choice(
            ["Rust", "GraphQL", "Kafka", "Spark", "Snowflake", "dbt", "Airflow"],
            size=int(rng.integers(0, 3)),
            replace=False,
        )
        if len(noise_skills):
            resume_text += " Additional: " + ", ".join(noise_skills) + "."

        data.append({
            'resume_text': resume_text,
            'category': category,
            'data_source': 'holdout' if is_holdout else 'train',
        })

    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset with {n_samples} resumes at {output_path}")
    print(f"  Holdout: {(df['data_source'] == 'holdout').sum()}, Train: {(df['data_source'] == 'train').sum()}")
    print(df['category'].value_counts())
    return output_path


if __name__ == "__main__":
    sample_path = create_sample_dataset()
    preparator = DataPreparator(sample_path)
    preparator.load_data()
    preparator.explore_data()
    preparator.clean_data()
    preparator.balance_classes()
    preparator.save_cleaned_data()
