import gradio as gr
import pandas as pd
from utils import fetch_news_data 


def gradio_interface(company_name, article_number):
    news_df_output = pd.DataFrame(columns=["Title", "Source"])
    json_summary = {}
    english_news_list = []
    hindi_news_list = []
    # hindi_news_text = None
    hindi_news_audio = None
    pie_chart = None
    bar_chart = None

    for result in fetch_news_data(company_name, int(article_number)):
        news_df_output = result.get("news_df_output", news_df_output)
        json_summary = result.get("json_summary", json_summary)
        english_news_list = result.get("english_news_list", english_news_list)
        hindi_news_list = result.get("hindi_news_list", hindi_news_list)
        # hindi_news_text = result.get("hindi_news_text", hindi_news_text)
        hindi_news_audio = result.get("hindi_news_audio", hindi_news_audio)
        pie_chart = result.get("pie_chart", pie_chart)
        bar_chart = result.get("bar_chart", bar_chart)

        yield news_df_output, json_summary, english_news_list, hindi_news_list, hindi_news_audio, pie_chart, bar_chart

with gr.Blocks(css=".btn-green { background-color: #2E7D32 !important; color: white !important; }") as interface:
    gr.Markdown("# Live Company News Analyzer")
    gr.Markdown("## A Project by Sara Nimje")
    gr.Markdown("Enter a company name to fetch news, sentiment analysis, and more.")

    with gr.Row():
        company_name_input = gr.Textbox(label="Company Name", placeholder="Enter company name")
        article_number_input = gr.Textbox(label="Number of Articles", placeholder="Enter number")

    with gr.Row():
        submit_btn = gr.Button("Submit", elem_classes=["btn-green"])
        clear_btn = gr.Button("Clear")

    with gr.Row():
        news_df_output = gr.Dataframe(label="News Articles", interactive=False)

    with gr.Row():
        json_summary_output = gr.JSON(label="JSON Summary")

    with gr.Row():
        english_news_output = gr.List(label="English News List")
        hindi_news_output = gr.List(label="Hindi News List")

    with gr.Row():
        # hindi_news_text_output = gr.Textbox(label="Hindi News Text", interactive=False)
        hindi_news_audio_output = gr.Audio(label="Hindi News Audio")

    with gr.Row():
        pie_chart_output = gr.Image(label="Sentiment Pie Chart")
        bar_chart_output = gr.Image(label="Sentiment Bar Chart")

    submit_event = submit_btn.click(
        gradio_interface,
        inputs=[company_name_input, article_number_input],
        outputs=[
            news_df_output,
            json_summary_output, 
            english_news_output, 
            hindi_news_output,  
            hindi_news_audio_output, 
            pie_chart_output, 
            bar_chart_output
        ]
    )

    company_name_input.submit(fn=gradio_interface, inputs=[company_name_input, article_number_input], outputs=[
        news_df_output,
        json_summary_output, 
        english_news_output, 
        hindi_news_output, 
        hindi_news_audio_output, 
        pie_chart_output, 
        bar_chart_output
    ])
    
    article_number_input.submit(fn=gradio_interface, inputs=[company_name_input, article_number_input], outputs=[
        news_df_output,
        json_summary_output, 
        english_news_output, 
        hindi_news_output, 
        hindi_news_audio_output, 
        pie_chart_output, 
        bar_chart_output
    ])

    clear_btn.click(
        lambda: ("", "", pd.DataFrame(), {}, "", "", None, None),
        inputs=[],
        outputs=[
            company_name_input, 
            article_number_input,
            news_df_output, 
            json_summary_output, 
            english_news_output, 
            hindi_news_output, 
            hindi_news_audio_output, 
            pie_chart_output, 
            bar_chart_output
        ]
    )

# launch app
if __name__ == "__main__":
    interface.launch()
