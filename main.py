import gradio as gr
from transformers import pipeline
# Create title, description and article strings
title = "SQUAD BERT Question Answering"
description = "'Langchain Demo With SQUAD'"
article = "Created at Pytorch"

model_checkpoint = "Chessmen/squad"
squad_task = pipeline("question-answering", model=model_checkpoint)

def predict(question, context):
    output = squad_task(question=question, context=context)["answer"]
    print(output)
    return output

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=[gr.Textbox(label="Type your string with your question:", lines=2),
                            gr.Textbox(label="Type your string with your context:", lines=2)], # what are the inputs?
                    outputs="text", # what are the outputs # our fn has two outputs, therefore we have two outputs
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=False) # generate a publically shareable URL?