# MuffinLM

> **MuffinLM Documentation**  
> *(Please read everything! and use only Elizabeth for now.)*

---
>---
> **UPDATE**
> I no longer update this repository in this account, but i do it on [2F AI](https://github.com/2F-AI)
>
>---
---

## **This project is licensed under the CC BY-NC 4.0 License.**
![CC BY-NC]( https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc.svg " ")

---

MuffinLM (also known as MLM) is a lightweight text-generation AI model designed for easy use and flexibility, a project coordinated by [FlameF0X](https://github.com/FlameF0X/) and [2F AI](https://github.com/2F-AI)

---

## Something

|                            **Model**                            |    **Type**     | **Parameters** |
|-----------------------------------------------------------------|-----------------|----------------|
|   [Muffin 2.7l](https://github.com/2F-AI/MuffinLM/tree/main)    | Text Generation |      5.8M      |
| [Muffin 2.8](https://github.com/2F-AI/MuffinLM/tree/chat-model) |    "Chatbot"    |      4.4M      |

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Available Versions](#available-versions)
3. [Datasets](#datasets)
4. [Saving Generated Text](#saving-generated-text)
5. [Version Coding](#version-coding)
6. [Requirements](#requirements)
7. [Enjoy!](#enjoy)

---

## Getting Started

To launch the current version of Muffin (Muffin V2.7l), simply run the following command:

```bash
python main.py
```

---

## Available Versions

Here are the available versions of MuffinLM:

|   **Model**   | Parameters |
|---------------|------------|
| Muffin v2.7l  |    5.8M    |
---

## Datasets

The `Snapshots` folder contains a `Datasets` folder with the following files:

- `dataset-5-large.txt` *(used to train Muffin v2.7l)*

## Saving Generated Text

When using Muffin v5.x, you will be prompted with:

```python
save_choice = input(">> Do you want to save the generated text? (yes/no/cancel/stop): ").strip().lower()
```

To save the generated text, type `yes`. The text will be saved to `SaveGeneratedText.txt`.

---

## Version Coding

**MAJOR.MINOR.CODE**

Muffin v2 is split into three sub-versions: Muffin v2.7f, Muffin v2.7c and Muffin v2.7l. The letter after V5 denotes the version type:

- **c**: Designed to be more casual.
- **f**: Designed to be more fancy, ideal for poetic text.
- **l**: Designed to be more large.

## Requirements

Muffin V2 and newer versions require PyTorch to run. You can install it with:

```bash
pip install torch
```

---

## Enjoy!

We hope you enjoy using MuffinLM!
```

### Key Changes Made:
1. **Added a Table of Contents**: For easy navigation.
2. **Improved Section Headings**: Made them more descriptive.
3. **Streamlined Information**: Organized the flow of information for clarity.
4. **Enhanced Formatting**: Used bullet points and bold for emphasis to make the document more visually appealing.
5. **Clarified Instructions**: Improved the wording in various sections for better understanding.
