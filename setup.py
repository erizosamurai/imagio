from setuptools import setup, find_packages

setup(
    name='imagio',
    version='0.1.0',
    description='Imagio: A CLI tool for semantic image search using CLIP and Faiss',
    author='Your Name',
    license='MIT',
    packages=find_packages(),  # includes semantic_search/
    install_requires=[
        'torch',
        'transformers',
        'faiss-cpu',
        'Pillow',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'imagio=semantic_search.cli:main'
        ]
    },
    include_package_data=True,
    python_requires='>=3.7',
)
