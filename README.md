# Project Decadence

## Intro

Decadence is a prototype of a self-learning AI-resistant automated Turing test (captcha). It is based on a simple game of finding common association to a bunch of words. The model uses a fully-connected weighted graph to store the "known" words as nodes and information about the direct associativity of each word pair as weights of corresponding edges.

The project employs the idea of a module-based algorithm, allowing one to build a captcha perfectly matching his unique requirements by combinig simple blocks and tuning a reasonable set of hyperparameters.

## Dependencies

```bash
apt update
apt install python3 -y
python3 -m pip install -U matplotlib networkx[default]
```

*(For* ***apt*** *don't forget your* `sudo` *and for* ***pip*** *consider using a* ***Python virtual environment*** *...)*

The version of python should be 3.5 or higher.

## Usage

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

## How does it work?

*During the following explanation we will often use the "human associativity" term. Human associativity is an abstract characteristic reflecting both the probability of some words being considered associatively close by a human and the probability of the same words **not** being considered associatively close by a bot. Human associativity should be raised if a person tends to bind the words with association and reduced if a bot makes the same decision.*

### Graph structure

As it was already mentioned, all the model's data is stored in a form a fully-connected weighted graph, where nodes are labeled with words, learned by the model. All the core scripts are written in Python, the graph maintainance is provided with the [NetworkX library](https://networkx.org/). Edge weights, physically stored in the graph database, are what we will from now on call "raw weights" – simple float numbers in range [-WEIGHT_LIMIT, WEIGHT_LIMIT], assigned to each existing edge. Raw weights are used to store the information about the words' human associativity in a simple and manageable form.

### Associativity metrics

Of course, measuring human associativity of a words couple by just looking up the raw weight of an edge, directly connecting the couple, would be too simple. First of all, when computing the human associativity score, we apply the [standard logistic function](https://en.wikipedia.org/wiki/Logistic_function) to all the raw edge weights we deal with – this decision will be explained later in the [Learning process](#learning-process) section. The results of such operation will be now referenced as the SL edge weight. Secondly, as word associations are partially transitive (if word A is strongly associated with word B and word B is strongly associated with word C, word A is at least considerably associated with word C) our model should be able to recognize indirect associations between the words. That is why to get the final associativity characteristic of a pair of words we use the [resistance distance](https://en.wikipedia.org/wiki/Resistance_distance) between the corresponding nodes, calculated over the SL-weighted edges of the whole graph. From now on when mentioning the "human associativity distance" we'll be talking about **this** value.

### Generating a riddle

"Riddle" here means just a set of words, that a tested user is offered to find a common association to. The possible approaches to generating this set differ and may (and should!) be combined with each other. By this moment we have come up with 2 separate ways to do it: we call them **random bunches** and **dense bunches**:

1. **Random bunches**. It is the simplest way to generate a set of words: we just pick several completely random graph nodes and show them to the tested user. It is easy to code and computationally cheap, but this method has an obvious major drawback: the chosen words in most cases will be hardly associated with each other. This means that finding a reasonable common association will often be impossible and users' anwers to such a riddle will be almost random. For this reason we do not consider random bunching an applicable practical way of implementing a captcha but we will use it to demonstrate the module-based structure of our algorithm.

2. **Dense bunches**. To generate one, we firstly pick a random word (node), name it the first (and initially the only) member of the bunch and then for several iterations we do the following: select some fixed amount of random unique nodes, for each one of them calculate the **mean** human associativity distance between it and the members of the bunch, add the sample with the lowest mean distance to the bunch. The resulting riddle word set is dense (internally human-associative) and so doesn't suffer the disadventages of the random bunch. Despite this, "dense" approach also has weak sides. First – the words in the bunch may be too strongly human-associated with each other, which will lead to any word that is well-human-associated with **only one** member of the bunch being considered well-human-associated with the **whole bunch** (transitivity, remember?). To compensate this effect we suggest "blocking" (setting the SL weight to infinity) the edges directly connecting the members of the bunch when computing the mean human associativity distance for some external sample – both while generating the bunch and while evaluating the answer to the riddle. From now on we will mean **this** (modified) procedure when mentioning "dense bunches". Another drawback of dense bunching is that by showing the user a set of well-human-associated words we provide him with information on how the human associativity looks like. This kind of data may be sufficient for training an AI capable of hacking out captcha.

### Making a verdict

The process of actually deciding who is a bot and who is a human is pretty simple. We take the user's input and follow one of the following scripts:

1. If the word is new to the model (there is no node in the graph labeled with this word) we add a node with a corresponding label to the graph, connect it with all other nodes with default-weighted edges, temporarily memorize the riddle and the user's input and regenerate the riddle.

2. If the model is familiar with this word, we calculate the mean human associativity distance between it and the members of the riddle (blocking internal riddle edges, if the word bunch is dense) and compare it with some predefined reference value.

### Learning process

To stay precise and efficient our model should learn. To make this happen right after deciding if the user a bot or a human we simply reweight the edges, directly connecting the word suggested by the user with the members of the riddle bunch. We do this by adding (subtracting) some predefined value to (from) the **raw** weight of these edges when the model believes the user is a bot (a human). And here comes the explanation why we introduced the SL weights: standard logistic function helps the human associativity distance to be a realistic non-linear smooth function of successful human-like associations recorded – meaning that at least its first derivative gets much closer to zero when the raw edge's weight approaches one of its limits.

It is important to mention that the learning process affects not only the answer word and the riddle word set that were actually used for making a verdict, but also all the word to word set connections suggested by the currently tested user and temporarily memorized by the model while adding new words (script 1 of [Making a verdict](#making-a-verdict) section).

### Module-based structure

As we already mentioned, both bunching techniques have some disadventages. We suppose that the key to coping with them (besides model's hyperparameters' fine-tuning, of course) is combining the techniques. For example, by making the user answer multiple riddles we can hide the information about the databases internal structure. Generating 3 randomly ordered riddles, 2 of which will be random bunches and 1 – a dense bunch, will probably do a great job. Besides that, we can use separate riddles for making a verdict and for training our module. Continuing our example: all three riddles may be used for learning but only the dense one – for deciding whether the user is a human being. Number of possible combinations is endless and we believe that exploring them is a key to creating a perfect self-training captcha.

*You can find an example of simple single-bunched models in scripts/decadence.py and an example of a 2-random-1-dense model in scripts/polydence.py (see the [Usage](#usage) section).*

## CLI commands reference

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
| quit | `> quit` | End the session, **save the current database state to the database/graph.gml file (!)**, close the CLI |

## Model's hyperparameters reference

*Hyperparameters can be tuned by changing the corresponding constants in the corresponding .py files.*

| Parameter | Role | Value type | Valid when using |
|:-:|:-|:-|:-|
| PATTERN | Used to switch a universal single-bunch model (e.g. scripts/decadence.py) between "single-random-bunch" and "single-dense-bunch" modes | **'single_dense'** or **'single_rand'** | universal single-bunch model |
| WEIGHT_ELASTICITY | Used to determine the least possible step of reweighting the edges (meaning the modification of the raw edge weight that will be used as an argument of standard logistic function when computing the distance between some pair of words) | positive **float**, significantly less than WEIGHT_LIMIT value | any model |
| WEIGHT_LIMIT | Used to determine the boundaries in which the raw edge weight may be variated: from -WEIGHT_LIMIT to WEIGHT_LIMIT | positive **float**, significantly greater than WEIGHT_ELASTICITY value | any model |
| WORD_SET_SIZE | Used to determine the number of words in a riddle, generated by the model after receiving a "get" request | positive **int** | any model |
| THRESH | Used to determine a threshold above (below) which the score (mean resistance distance between an answer word and riddle words, calculated over standard logistic function values, applied to raw edge weights) will be considered to be machine- (human-) like | positive **float** belonging to (0, 1) | any model |
| DEFAULT_EDGE_WEIGHT | Used to determine the default raw edge weight, that is initially assigned to all the edges attached to a newly added node (via "insert" or "post" with a word that the model is not familoar with) | **float** belonging to [-WEIGHT_LIMIT, WEIGHT_LIMIT] | any model |
| HEURISTIC_RATE | Used to calculate the maximum SL weight of the edge, above which it will be considered too heavy and excluded from the graph while calculating the resistance distance between two nodes to reduce the computational complexity: this limit equals HEURISTIC_RATE * <SL weight of the direct edge between those nodes> | positive **float** | any model |
| DENSE_SAMPLING_RATE | Used to determine the number of nodes in the sample, from which the closest one will be chosen in the process of generating a dense bunch: greater sampling rate => less random bunches + more computationally expensive generation routine | positive **int** | any model using dense bunches |
