from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from extensions import db
from database.models import Prediction
import json

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
def landing():
    from flask_login import current_user
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    return render_template("landing.html")

@dashboard_bp.route("/dashboard")
@login_required
def index():
    counts     = current_user.prediction_counts()
    recents    = current_user.recent_predictions(5)
    activity   = current_user.weekly_activity()
    total      = sum(counts.values())
    last_pred  = recents[0] if recents else None
    return render_template("dashboard/index.html",
                           counts=counts,
                           recents=recents,
                           activity=activity,
                           total=total,
                           last_pred=last_pred)

@dashboard_bp.route("/profile")
@login_required
def profile():
    counts  = current_user.prediction_counts()
    total   = sum(counts.values())
    recents = current_user.recent_predictions(10)
    return render_template("profile/index.html",
                           counts=counts, total=total, recents=recents)
