from utils.logger import get_logger

logger = get_logger(__name__)

class Metrics:

    def __init__(self):
        self.questions = 0
        self.unknown = 0
        self.leads = 0

    def record_question(self):
        self.questions += 1

    def record_unknown(self):
        self.unknown += 1

    def record_lead(self):        
        self.leads += 1

    def summary(self):
        logger.info(
            f"questions={self.questions} "
            f"unknown={self.unknown} "
            f"leads={self.leads}"
        )


metrics = Metrics()