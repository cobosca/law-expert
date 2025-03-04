from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import time
import openai

import main_neo as x


app = Flask(__name__)
app.config['SECRET_KEY'] = 'baigislepeni'
socket = SocketIO(app)

gatekeeper = False # by default the gates rremain open

@socket.on('message')
def handle_message(message):
    global gatekeeper
    print('Received message:', message)
    
    usr_question = message
    df = x.df
    model = x.GPT_MODEL
    token_budget = 4096 - 500
    
    question_to_gpt = x.query_message(usr_question, df, model=model, token_budget=token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about a specific law in latvia. You act and speak professionally as would a trained lawyer. You always speak Latvian as your default language, unless asked a question in a different language"},
        {"role": "user", "content": question_to_gpt},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True, # Das ist ein gross Fluss
    )
    for chunk in response:
        try:
            response_chunk = str(chunk.choices[0].delta.content)
        except AttributeError:
            continue
        time.sleep(0.001)
        socket.emit('message', response_chunk)
        print(response_chunk)
        # Broadcast the message to all connected clients
        
    gatekeeper = True # close and guard the gates. Dont let pass through
    
@socket.on('gatekeeper')
def handle_question(g_question):
    global gatekeeper
    print(g_question)
    if gatekeeper:
        print("Yes, the gates are now locked")
        socket.emit('gatekeeper')
        gatekeeper = False
    else:
        print("No, the gates are open for now! You shall pass!")

@app.route("/process_string", methods=['POST'])
def process_string():
    usr_input = request.json.get('data') #panem to izvelk to data no json no js
    result = usr_input
    print(result)
    return jsonify(result=result)

@app.route('/')
def home():
    return render_template('index.html')


app.static_folder = 'static'

if __name__ == "__main__":
    socket.run(app, host="0.0.0.0", port=8000, debug=True)