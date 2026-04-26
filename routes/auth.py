from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from database.models import User
from datetime import datetime

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=remember)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard.index"))
        error = "Invalid email or password."
    return render_template("auth/login.html", error=error)

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    errors = {}
    form_data = {}
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        form_data = {"name": name, "email": email}

        if not name or len(name) < 2:
            errors["name"] = "Name must be at least 2 characters."
        if not email or "@" not in email:
            errors["email"] = "Please enter a valid email address."
        if len(password) < 8:
            errors["password"] = "Password must be at least 8 characters."
        if password != confirm:
            errors["confirm"] = "Passwords do not match."
        if User.query.filter_by(email=email).first():
            errors["email"] = "This email is already registered."

        if not errors:
            user = User(name=name, email=email, last_login=datetime.utcnow())
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Welcome to MedDiagno! Your account has been created.", "success")
            return redirect(url_for("dashboard.index"))
    return render_template("auth/register.html", errors=errors, form_data=form_data)

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
