import gradio as gr
import requests
import json
import PyPDF2
from io import BytesIO
import os

# Global variable to store extracted resume text
current_resume_text = ""

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def process_resume(pdf_file):
    """Process the uploaded resume and store text globally"""
    global current_resume_text

    if not pdf_file:
        return "Please upload a PDF file.", ""

    try:
        current_resume_text = extract_text_from_pdf(pdf_file)
        if not current_resume_text:
            return "No text could be extracted from the PDF. Please ensure the PDF contains readable text.", ""

        success_message = f"✅ Resume processed successfully! ({len(current_resume_text)} characters extracted)\n\nYou can now ask questions about this resume in the chat."
        return success_message, ""

    except Exception as e:
        current_resume_text = ""
        return f"Error processing resume: {str(e)}", ""


def answer_question(question, chat_history):
    """Answer questions about the uploaded resume"""
    global current_resume_text

    if not current_resume_text:
        response = "❌ Please upload and process a resume first before asking questions."
        chat_history.append([question, response])
        return chat_history, ""

    if not question.strip():
        response = "Please enter a question about the resume."
        chat_history.append([question, response])
        return chat_history, ""

    try:
        prompt = f"""
        Based on the following resume content, please answer the user's question accurately and concisely:
        Resume Content:
        {current_resume_text}
        User Question: {question}
        Please provide a clear, specific answer based only on the information available in the resume. If the information is not available in the resume, please state that clearly.
        """

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about resumes. Base your answers strictly on the resume content provided. Be concise but thorough."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )

        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
        else:
            answer = f"❌ API Error: {response.text}"

        chat_history.append([question, answer])
        return chat_history, ""

    except Exception as e:
        response = f"❌ Error: {str(e)}"
        chat_history.append([question, response])
        return chat_history, ""


def clear_chat():
    return []


def get_sample_questions():
    return [
        "What are the key technical skills mentioned?",
        "What is their educational background?",
        "What certifications do they have?",
        "Rate this resume on a scale of 1-10"
    ]

# Gradio UI
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="green")) as demo:
        gr.Markdown("<h1 style='text-align: center; color: green;'>ResumAI</h1>")
        gr.Markdown("<p style='text-align: center; color: white;'>Upload a resume (PDF) and ask specific questions about the candidate's skills, experience, and qualifications.</p>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Upload Resume")
                pdf_input = gr.File(label="Upload PDF Resume", file_types=[".pdf"])
                process_button = gr.Button("Process Resume", variant="primary")
                status_text = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("### 💡 Sample Questions")
                sample_questions = get_sample_questions()
                for question in sample_questions:
                    gr.Markdown(f"• {question}")

            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions About the Resume")
                chatbot = gr.Chatbot(label="Q&A Chat", height=570)
                with gr.Row():
                    question_input = gr.Textbox(label="Your Question", placeholder="Ask anything about the resume...", scale=4)
                    ask_button = gr.Button("Ask", variant="primary", scale=1)
                with gr.Row():
                    clear_button = gr.Button("Clear Chat", variant="secondary")

        process_button.click(fn=process_resume, inputs=[pdf_input], outputs=[status_text, question_input])
        ask_button.click(fn=answer_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
        question_input.submit(fn=answer_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
        clear_button.click(fn=clear_chat, outputs=[chatbot])

    return demo

if __name__ == "__main__":
    print("Starting Resume Q&A Assistant with DeepSeek...")
    demo = create_ui()
    demo.launch()
