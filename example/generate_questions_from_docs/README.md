# Custom Prompt Data

This is an example on how to generate questions with a custom prompt. It uses the `gpt-4.1` model to generate developer style questions from the huggingface transformers trainer documentation

## download the data files:
```bash
curl https://huggingface.co/docs/transformers/trainer -o trainer.html
curl https://huggingface.co/docs/transformers/training -o ft.html
curl https://huggingface.co/docs/transformers/optimizers -o optimizers.html
```

## then run

```bash
# set an OPENAI API KEY
export OPENAI_API_KEY=
# run the script
yourbench run example/generate_questions_from_docs/config.yaml
```

The expected result from this run can be found at the following huggingface dataset: [yourbench/yourbench-custom-prompts-example](https://huggingface.co/datasets/yourbench/yourbench-custom-prompts-example/viewer/single_shot_questions)