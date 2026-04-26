import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask
from config import Config
from extensions import mongo, login_manager

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    mongo.init_app(app)
    login_manager.init_app(app)

    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.predictions import pred_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(pred_bp)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    with app.app_context():
        db.create_all()

    @app.template_filter("timeago")
    def timeago_filter(dt):
        from datetime import datetime, timezone
        if dt is None:
            return "never"
        now   = datetime.utcnow()
        delta = now - dt
        s = int(delta.total_seconds())
        if s < 60:     return "just now"
        if s < 3600:   return f"{s//60}m ago"
        if s < 86400:  return f"{s//3600}h ago"
        return f"{s//86400}d ago"

    @app.template_filter("pct")
    def pct_filter(val):
        return f"{val:.1f}%"

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=5000)
