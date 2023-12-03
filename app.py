import gradio as gr

# ------------- Global Variables
DAYS = [
    "Su, 26.11.2023 ",
    "Mo, 27.11.2023",
    "Tu, 28.11.2023",
    "We, 29.11.2023",
    "Th, 30.11.2023",
    "Fr, 01.12.2023",
    "Sa, 02.12.2023",   
]

TITLE = "Travel Planner" # TODO: Change title as we need

# ------------- Main App
with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Default(primary_hue=gr.themes.colors.red).set(slider_color="#ff0000"),
    css=".travel-block {max-width: 50%; margin: 0 auto}",
) as demo:

    with gr.Column(elem_classes="travel-block") as left:
        # Input
        with gr.Row() as header:
                gr.Markdown(
                    """
                    ## 🚀 Search For You Next Trip
                    """,
                )

        A = gr.Textbox(label="Start", placeholder="Lausanne")
        B = gr.Textbox(label="Destination", placeholder="Basel")

        with gr.Row() as metainfo:
            limit = gr.Slider(minimum=1, maximum=5, step=1, label="Limit", interactive=True, value=3)
            days = gr.Dropdown(DAYS, label="Day", value=DAYS[-1])
            time = gr.Textbox(label="Time", placeholder="hh:mm")

        submit = gr.Button(value="Search", variant="primary")

        # Output
        with gr.Column(visible=False) as output:
            route = gr.Markdown()

        def search(A, B, limit, day, time):
            """Search for the best route from A to B"""
            md = f"""
            ---

            🚀 **{A}** to 🏁 **{B}** at ⌚ **{time}** on 📅 **{day}**
            
            ---
            """

            return {
                output: gr.Column(visible=True),
                route: md
            }

        submit.click(
            search,
            inputs=[A, B, limit, days, time],
            outputs=[route, output]
        )

demo.launch()
