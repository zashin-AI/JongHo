from flask import Flask

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return '''
    <h1> 이건 너목보 </h1>
    <p> 이건 돈.. 돈.. 은행 </p>
    <p> 내일 오전 10시까지 정하자! </p>
    <a href="https://naver.com">Naver 홈페이지 바로가기</a>
    '''

@app.route('/user/<user_name>/<int:user_id>')
def user(user_name, user_id):
    return f'Hello, {user_name}({user_id})!'

if __name__ == '__main__':
    app.run(debug=True)