from flask import Flask, jsonify
from app.routers.predictions import predictions_bp


def create_app():
    app = Flask(__name__)

    # 블루프린트 등록
    app.register_blueprint(predictions_bp, url_prefix='/api/v1')

    @app.route('/')
    def home():
        return jsonify({
            "message": "Welcome to APT Auction AI API",
            "version": "1.0",
            "endpoints": {
                "home": "/",
                "health": "/health"
            }
        })

    @app.route('/health')
    def health_check():
        return {"status": "healthy"}

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
