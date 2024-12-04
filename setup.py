'''
Date: 2024-12-04 16:54:16
LastEditors: yangyehan 1958944515@qq.com
LastEditTime: 2024-12-04 17:12:53
FilePath: /herberta/setup.py
Description: 
'''
from setuptools import setup, find_packages

setup(
    name="herberta",  # 包名
    version="0.1.0",
    author="XiaoEn",
    author_email="1958944515@qq.com",
    description="A Python package for converting texts to embeddings using pretrained models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/15392778677/herberta",  # 替换为您的仓库地址
    packages=find_packages(),  # 自动发现子包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.9",
        "transformers>=4.0",
    ],
)