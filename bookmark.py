import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS  # Import Flask-CORS
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Email, Mail, To

# Load the environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)


@app.route("/send-url", methods=["POST"])
def send_url_email():
    data = request.get_json()
    print("Received data:", data)
    # Validate that required fields are present
    if not data or "urls" not in data or "recipient_email" not in data:
        return (
            jsonify({"error": "Missing required fields: urls (list), recipient_email"}),
            400,
        )

    urls = data["urls"]
    # Validate that urls is a list
    if not isinstance(urls, list):
        return jsonify({"error": "urls must be a list"}), 400

    recipient_email = data["recipient_email"]

    # Create HTML content with all URLs
    urls_html = "".join([f"<p><a href='{url}'>{url}</a></p>" for url in urls])

    # Create the email message using SendGrid
    message = Mail(
        from_email=Email(os.environ.get("SENDGRID_EMAIL")),
        to_emails=To(recipient_email),
        subject="Your Requested URLs",
        html_content=f"<p>Here are the URLs you provided:</p>{urls_html}",
    )
    try:
        sg = SendGridAPIClient(os.environ.get("SENDGRID_API_KEY"))
        response = sg.send(message)
        return (
            jsonify(
                {
                    "message": "Email sent successfully!",
                    "status_code": response.status_code,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app (debug mode is enabled for development)
    app.run(host="0.0.0.0", port=5001, debug=True)
