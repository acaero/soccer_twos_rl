## Setup

1. Use `Python 3.9` going forward.
2. Initialize a virtual environment using `<PATH_PYTHON_3.9>/python.exe -m venv .venv`. **Note: `virtualenv` will fail due to legacy requirements for the `setup.py`.**
3. (Optionally) Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) for GPU training
4. Run `pip install --no-cache-dir -r requirements.txt ` to install all requirements.
5. Run `tensorboard --logdir src\runs\out\logs\tensorboard`

## Pre-existing environment

The code for the environment can be found [here](https://github.com/bryanoliveira/soccer-twos-env).

An overview of MARL (Multi Agent Reinforcement Learning) research [here](https://github.com/LantaoYu/MARL-Papers).

## How to run?
**NOTE: Generally it is advised to run each file via a module call so that cross project imports works. E.g. `python -m src.runs.arena`**

The repo is divided into the section `src.agent`, `src.runs` and `src.visualization`. In the first you can find our self implemented versions of algorithms, the second folder then contains runners to train these algorithms. The visualization then continues with our pre-trained models and also the `arena.py` file. In here you can easily select agents to play a match against each other and watch them play.

During the training there will be logs captured. Therefore within the `src.runs` folder there will appear an `out` folder which contains the saved model at the end and also a tensorboard log file next to a plain text log file. To monitor your training execute this in the activated virtual environment: `tensorboard --logdir=src/runs/out/logs/tensorboard`

In the notebooks folder you can find scripts to generate pretty visualized logs. However you do need some logging data first.
## Requirements

### Thema:

- Beliebiges Reinforcement-Problem (Brettspiel, Kartenspiel, Computerspiel,
  Robotiksimulator...)
- Anwendung von Methoden aus der Vorlesung
- Auseinandersetzung mit dem Thema wichtig
- Keine Erwartung von bahnbrechenden Ergebnissen

### Umfang:

- Programmcode
- Schriftliche Ausarbeitung (5-10 Seiten) (Einführung -> Methoden -> Anwendung -> Abschluss)
- Abschlusspräsentation (20 Minuten, 8 Folien, jeder soll Vortragen)

## Insights

- Die Aktionsraum eines jeden Spielers ist in drei Dimensionen aufgeteilt `[<Vorgehen-Zurückgehen>, <Rechtsgehen-Linksgehen>, <Linksdrehen-Rechtsdrehen>]`
- Dabei muss jeder Wert eine Ganzzahl im Bereich `[1, 2]` sein.
  Wobei `1` das erste Element des jeweiligen Wertes bezeichnet und `2` das zweite. E.g.

- `[1, 0, 0]` -> geht nur Vorwärts,

- `[1, 2, 0]` -> geht Vorwärts und nach Links. und

- `[2, 2, 2]` -> geht Rückwärts, nach Links und dreht sich Rechts.

- Die Aktionen werden dabei während eines Runs immer für alle Spieler zeitgleich definiert, das übergeben dictionary sieht dabei wie folgt aus:

- `action={
    0: [0, 0, 0], # Blau hinten
    1: [0, 0, 0], # Blau vorne
    2: [0, 0, 0],# Orange hinten
    3: [0, 0, 0],# Orange vorne
}`

- Jedes game wird für entweder 1000 Iterationen, oder bis zu einem Tor gespielt
