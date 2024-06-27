import gradio as gr

def read():
    return 'sth'

demo = gr.ChatInterface(
    fn=read,
    input=[],
    output=["text"]
)

demo.launch()