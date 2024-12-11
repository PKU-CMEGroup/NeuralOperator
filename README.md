## Code Structure


NeuralOperator/

├── baselines/
│   ├── __init__.py
│   ├── fno.py
│   ├── geofno.py
│   ├── mgno.py
│   ├── pit.py

├── utility/
│   ├── __init__.py
│   ├── adam.py
│   ├── losses.py
│   ├── normalizer.py

├── pcno/
│   ├── __init__.py
│   ├── pcno.py
│   ├── geo_utility.py

├── tests/
│   ├── __init__.py
│   ├── pcno_test.py

├── scripts/
│   ├── (various test script folders)

├── data/
│   ├── (various data folders)



Folder and file names are all lower case.

Each time, when you update pcno, run pcno_test
To run tests, in NeuralOperator/ folder
    python -m tests.pcno_test