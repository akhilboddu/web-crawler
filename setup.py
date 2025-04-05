from setuptools import setup, find_packages

setup(
    name="web-crawler",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "playwright",
        "beautifulsoup4",
        "python-dotenv",
        "aiohttp",
        "fastapi",
        "uvicorn",
    ],
) 