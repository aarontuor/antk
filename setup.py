try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(name='antk',

      version=0.3,
      description='Automated Neural-graph Toolkit: A Tensorflow wrapper for '
                  'common deep learning tasks and rapid development of innovative'
                  'models. Developed at Hutch Research, Western Washington University.'
                  'Support for multiple input and output neural network graphs. '
                  'Great visualizations and extensively documented interface. '
                  'Great tool for exploring tensorflow functionality and deep learning fundamentals.',
      url='http://aarontuor.xyz',
      author='Aaron Tuor',
      author_email='tuora@students.wwu.edu',
      license='none',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['scipy', 'numpy'],
      classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Text Processing :: Linguistic'
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        'Documentation :: Sphinx'],
    keywords=[
        'Deep Learning',
        'Neural Networks',
        'TensorFlow',
        'Machine Learning',
        'Western Washington University',
        'Recommender Systems'])