# NLP First-experience Lab

**Hands-on First Experience of Natural Language Processing**

Authors:
- [Li Peng-Hsuan 李朋軒](https://jacobvsdanniel.github.io) (jacobvsdanniel [at] gmail.com)

## Overview

- Solve Natural Language Processing (NLP) tasks
- Apply Large Language Models (LLM)
- Intended for beginners
  - The source codes are short: be expected to understand them fully.
  - Be expected to develop your own program.
  - No GPU requirements or machine learning model engineering: we'll be using online services.

## Environment

- Operating system
  - OS-independent
- System apps
  - Python 3.9.5+
- Python packages
  - See *requirements.txt*

## Lab 1

Prompting Large Language Models (LLM) to solve Natural Language Processing (NLP) tasks

### Lab 1.0. Preparation

#### Step 1. These are the relevant files:

- *lab1_nlp_tasks.py*
- *lab1_config.json*
- *corpora/news/*
- *corpora/prompt/*

#### Step 2. These are the core concepts:
- Tasks
  - Named Entity Recognition (NER)
  - Relation Extraction (REL)
  - Summarization (SUM)
- LLM prompting methods
  - Instruction prompting
  - In-context learning
- Languages
  - English (en)
  - Chinese (zh)

#### Step 3. [Create your OpenAI API Key](https://platform.openai.com/account/api-keys/)

It is a string that looks like "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".

#### Step 4. [Fund your OpenAI account](https://platform.openai.com/account/billing/)

$5 should be sufficient to complete all the labs.

If you had trouble funding your account:
- Plan B
  - You can skip Step 3 and 4
  - And try to use the free [website](https://chat.openai.com/) to get results for all the labs.
  - WARNING: A lot of manual labor ahead!
- Plan C
  - You can skip Step 3 and 4
  - And try to download and use a smaller LLM, eg. [this](https://huggingface.co/docs/transformers/main/model_doc/llama2) and [that](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM), locally on your own GPU.
  - WARNING: Humongous engineering work ahead! And performance is not guaranteed!

#### Step 5. Check *lab1_config.json*

Modify it as you wish.

#### Step 6. Run *lab1_nlp_tasks.py*

For example, using linux command line, run
```bash
python lab1_nlp_tasks.py
```

#### Step 7. Check results in *lab1_output/*

For example, in *lab1_output/ner__instruction__en__Russo-Ukrainian_War.txt*:
```
Extract entities of persons, locations, organizations from the article.

Article:
The Russo-Ukrainian War is an ongoing international conflict...

####################################################################################################
# OUTPUT
####################################################################################################

1. Viktor Yanukovych (person)
...
```

The results may vary for each different runs.

### Lab 1.1. Solve existing tasks

#### Objectives

Solve all tasks (NER, REL, SUM) in both languanges (English, Chinese) with both prompting methods (instruction prompting, in-context learning).
- The English prompts is already there; you only need to get output results.

#### Submission

- Your prompt files.
- Your output results.
- (Lab 1.1 constitutes ~20% of the labs.)

### Lab 1.2. Solve new tasks

#### Objectives

Solve tasks beyond NER, REL, SUM
- For at least 2 tasks
  - At least one of them must be a well known NLP task.
  - For inspirations: [CKIP CoreNLP](https://ckip.iis.sinica.edu.tw/service/corenlp/)
  - For inspirations: [Stanford CoreNLP](https://corenlp.run/)
  - For inspirations: [Papers With Code - NLP](https://paperswithcode.com/area/natural-language-processing/)
  - For inspirations: [LDC Catalog](https://catalog.ldc.upenn.edu/)
- In at least 2 languages
  - One of them must be English
  - The other may be Chinese or another language
- Using multiple prompting methods
  - Instruction prompting: required
  - In-context learning: required
  - Feel free to also use other methods.

#### Submission

- Your prompt files
- Your output results
- (Lab 1.2 constitutes ~20% of the labs.)

## Lab 2

Using LLM train of thought to achieve complicated behavior

### Lab 2.0. Preparation

#### Step 1. These are the relevant files:

- *lab2_train_of_thought.py*
- *lab2_config.json*

#### Step 2. These are the core concepts:

- Parse LLM output to structured data
  - To use them programmatically beyond chatting with humans.
- Make LLM digest its own outputs
  - To get explanations, spot errors, edit results, do follow-up tasks, etc.

### Lab 2.1. A Druid's Journey

#### Objectives

Play the **修行的德魯伊(A Druid's Journey, 2023)** game
- Run *lab2_train_of_thought.py*
- Achieve the game objectives
- Check results in *lab2_output/*

#### Submission

- Your final game states and output files
- (Lab 2.1 constitutes ~20% of the labs.)

### Lab 2.2. Your own game

#### Objectives

Create your own game
- It may be a text adventure game or be in other forms of your wish.
- It must use multiple LLM tasks.
- It must require LLM to digest its own outputs.
- It must create an enjoyable review for each game run.

#### Submission

- Your game
  - Program
  - Data
  - Manual (if needed)
- Runs of your game
  - The final game states and output files of at least one full run
- A report
  - Game description
  - Details on the LLM tasks applied
  - Your discovery
- (Lab 2.2 constitutes ~40% of the labs.)

## LICENSE

Copyright © 2023 [Li Peng-Hsuan 李朋軒](https://jacobvsdanniel.github.io)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
