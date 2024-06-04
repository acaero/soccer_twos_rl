## Setup

1. Use `Python 3.9.1` going forward.
2. Initialize virtual environment via `<PATH_PYTON_3.9>/python.exe -m venv .venv` (note: `virtualenv` will fail due to legacy requirementy for the `setup.py`). 
3. Use `pip install --no-cache-dir -r requirements.txt ` to install all requirements.
4. Start Program via `python main.py`

## Inspiration

Inspiration code can be found [here](https://github.com/bryanoliveira/soccer-twos-env).

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
