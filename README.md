## Project Decadence

### Description

Decadence is a prototype of a self-learning AI-resistant automated Turing test (or just captcha). It is based on a simple game of finding common association to a bunch of words. The model uses a fully-connected weighted graph to store its "known dictionary", including the information about the associativity of each pair of words, encoded as the weight of the edges between the corresponding node pairs. Associativity score itself is computed using the [resistance distance](https://en.wikipedia.org/wiki/Resistance_distance), which makes it possible to take the associations' transitivity into account, and the [standard logistic function](https://en.wikipedia.org/wiki/Logistic_function), helping to consider the direct associativity as a realistic non-linear smooth function of number of association cases. As a part of the model's learning process, the graph's edges are being continually reweighted according to the answers of the users identified as humans or bots. 

The core scripts are written in Python, all the graph maintainance was provided with the [NetworkX library](https://networkx.org/).

### Dependencies

```bash
apt update
apt install python3 -y
python -m pip install -U matplotlib networkx[default]
```

*(For* ***apt*** *don't forget your* `sudo` *and for* ***pip*** *consider using a* ***Python virtual environment*** *...)*

### How-to-use

To launch a single-bunch model's CLI:

```bash
python3 scripts/decadence.py
```

To launch a 2-random-1-dense model's CLI:

```bash
python3 scripts/polydence.py
```

Please be aware that for correct models' functioning by the launch moment database/graph.gml should exist and should describe a graph with at least 1 node and artifacts/batch.dat should exist and should contain a single record of format "0 x" where x belongs to {0, 1, 2} set. Default single-node database can be generated using `python3 support/gen_default_graph.py` (be careful, it erases all the data currently stored in the database/graph.gml!) and a suitable batch.dat file can be generated using `python3 support/batch_init.py`.

*Note that the database/graph.png has ~100 pre-learned samples right out of the box, so it is not necessary to recreate the empty database before trying the model.*

### CLI commands reference

| Command | Usage | Purpose |
|:-:|:-|:-|
| test | `> test` | Switch the model to the test mode, forcing it to determine whether you are a bot or a human using only the data stored in the database |
| learn | `> learn` | Switch the model to the learning mode, allowing it to only process the input, manually labeled as human's or bot's one |
| get | `> get` | Mimic an http server's "get" request, request a riddle (a set of words to find a common association with) |
| post | `> post [hum/mac] <answer_word>` | Mimic an http server's "post" request, deliver a word, answering the riddle, labeled as "human" or "machine" if learning mode is enabled |
| insert | `> insert <word_1> <word_2> ...` | Manually import a bunch of words to the database, setting **all** their edges' weights to default value |
| remove | `> remove <word_1> <word_2> ...` | Manually remove a bunch of words from the database, forgetting all the associativity data related to these words |
| stat | `> stat` | Print a statistical report on the database's current state (currently the report only contains the number of nodes) |
| print | `> print` | Draw a visualization of the graph in the png/graph.png file |
| save | `> save` | Save the current database state to the database/graph.gml file |
| quit | `> quit` | End the session, save the current database state to the database/graph.gml file (!), close the CLI |

### Model's hyperparameters reference

*Hyperparameters can be tuned by changing the corresponding constants in the corresponding .py files.*

| Parameter | Role | Value type | Valid when using |
|:-:|:-|:-|:-|
| PATTERN | Used to switch a universal single-bunch model (e.g. scripts/decadence.py) between "single-random-bunch" and "single-dense-bunch" modes | **'single_dense'** or **'single_rand'** | universal single-bunch model |
| WEIGHT_ELASTICITY | Used to determine the least step of reweighting the edges (meaning the modification of the raw edge weight that will be used as an argument of standard logistic function when computing the distance between some pair of words) | positive **float**, significantly less than WEIGHT_LIMIT value | any model |
| WEIGHT_LIMIT | Used to determine the boundaries in which the raw edge weight may be variated: from -WEIGHT_LIMIT to WEIGHT_LIMIT | positive **float**, significantly greater than WEIGHT_ELASTICITY value | any model |
| WORD_SET_SIZE | Used to determine the number of words in a riddle, generated by the model after receiving a "get" request | positive **int** | any model |
| THRESH | Used to determine a threshold above (below) which the score (mean resistance distance between an answer word and riddle words, calculated over standard logistic function values, applied to raw edge weights) will be considered to be machine- (human-) like | positive **float** belonging to (0, 1) | any model |
| DEFAULT_EDGE_WEIGHT | Used to determine the default raw edge weight, that is initially assigned to all the edges attached to a newly added node (via "insert" or "post" with a word that the model is not familoar with) | **float** belonging to [-WEIGHT_LIMIT, WEIGHT_LIMIT] | any model |
| HEURISTIC_RATE | Used to calculate the maximum standard-logistic-function weight of the edge, above which it will be considered too heavy and excluded from the graph while calculating the resistance distance between two nodes to reduce the computational complexity: this limit equals HEURISTIC_RATE * <standard-logistic-function weight of the direct edge between those nodes> | positive **float** | any model |
| DENSE_SAMPLING_RATE | Used to determine the number of nodes in the sample, from which the closest one will be chosen in the process of generating a dense bunch: greater sampling rate => less random bunches + more computationally expensive generation routine | positive **int** | any model using dense bunches |

### Contacts

#### Korney Ivanishin, author of the project

E-mail: korney059@gmail.com,
GitHub: [quaiion](https://github.com/quaiion), Telegram: @quaiion
