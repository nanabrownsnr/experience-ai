"""
Setup file for ExperienceAI package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="experience-ai",
    version="0.3.2",
    author="nana brown",  # Update with your name
    author_email="nanabrownsnr@gmail.com",  # Update with your email
    description="Build AI system prompts that evolve through user interactions with intelligent LLM-based message classification and comprehensive preference learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nanabrown/experience-ai",
    project_urls={
        "Bug Tracker": "https://github.com/nanabrownsnr/experience-ai/issues",
        "Documentation": "https://github.com/nanabrownsnr/experience-ai#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No required dependencies - LLM clients are optional
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed
        ],
    },
)
