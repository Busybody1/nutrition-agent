from setuptools import find_packages, setup
import os

# Try to read README.md, fallback to empty string if not found
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Shared library for AI Agent Framework"

setup(
    name="shared",
    version="0.1.0",
    author="AI Agent Framework Team",
    author_email="team@fitness-ai.com",
    description="Shared library for AI Agent Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.5.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=5.0.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
    },
)
