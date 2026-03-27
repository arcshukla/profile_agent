from services.pushover_service import PushoverService


class Notifier:

    def __init__(self):
        self.provider = PushoverService()

    def notify_lead(self, name, email, session_id=""):

        msg = f"Lead captured [{session_id}]\n{name}\n{email}"

        self.provider.send(msg)

    def notify_unknown(self, question, session_id=""):

        msg = f"Unknown question [{session_id}]\n{question}"

        self.provider.send(msg)

    def notify_error(self, error_type, details, session_id=""):

        msg = f"External error [{session_id}]: {error_type}\n{details}"

        self.provider.send(msg)