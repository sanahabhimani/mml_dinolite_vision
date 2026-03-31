import smtplib
from email.message import EmailMessage


def send_email(
    subject,
    body,
    to_email,
    from_email,
    app_password,
    attachment_path=None,
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    # attach image if provided
    if attachment_path is not None:
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = attachment_path.split("/")[-1]

        msg.add_attachment(
            file_data,
            maintype="image",
            subtype="png",
            filename=file_name,
        )

    # send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(from_email, app_password)
        smtp.send_message(msg)


