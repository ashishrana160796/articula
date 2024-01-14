import dill as pickle

from evaluator.feature_generator import gb_mle_combined
from common.ghostbusters_featurize import normalize
from spoofer.baseline_text_spoofer import BaselineTextSpoofer

# Load model
model = pickle.load(open("models/comb_model_rf", "rb"))
mu = pickle.load(open("models/comb_mu", "rb"))
sigma = pickle.load(open("models/comb_sigma", "rb"))

def fool_detector_text(text):

    features = gb_mle_combined(text)
    normalized_features = (features - mu) / sigma

    preds = model.predict_proba(normalized_features.reshape(-1, 1).T)[:, 1]

    if preds >= 0.5:
        print("The text is AI Generated!!! Spoofing...")
        text_spoofer = BaselineTextSpoofer(text, add_info_mutation=True)
        spoofed_text = text_spoofer.spoof_text()
        print(spoofed_text)
    else:
        print("Text is Human Written!")


if __name__ == '__main__':
    input_text = ["Sehr geehrte Damen und Herren, "
            ""
            "mit großem Interesse bewerbe ich mich für die Stelle als Fachangestellter im Bürohandel in Ihrem "
            "Unternehmen. Aufgrund meiner fundierten Erfahrung im Bürohandel und meiner Begeisterung für die "
            "Organisation und Verwaltung von Büromaterialien, sehe ich diese Position als eine hervorragende Gelegenheit, "
            "meine Fähigkeiten und Fachkenntnisse einzubringen."
            ""
            "Während meiner bisherigen beruflichen Laufbahn habe ich umfangreiche Kenntnisse im Einkauf und in der "
            "Lagerverwaltung von Bürobedarf erworben. Ich bin vertraut mit verschiedenen Bürosoftwareanwendungen und "
            "habe eine ausgezeichnete Fähigkeit zur Kundenbetreuung entwickelt. Zudem bin ich äußerst organisiert, "
            "detailorientiert und arbeite effizient, um sicherzustellen, dass die Büroprodukte stets verfügbar "
            "sind und den Bedürfnissen der Kunden gerecht werden."
            ""
            "Meine Leidenschaft für den Bürohandel und mein Engagement für exzellenten Kundenservice haben mich dazu "
            "motiviert, stets auf dem neuesten Stand der Entwicklungen in der Branche zu bleiben. Ich bin überzeugt, "
            "dass ich eine wertvolle Ergänzung für Ihr Team sein kann und freue mich darauf, dazu beizutragen, "
            "die hohen Standards Ihres Unternehmens aufrechtzuerhalten."
            ""
            "Vielen Dank für Ihre Zeit und die Berücksichtigung meiner Bewerbung. Ich stehe Ihnen gerne für ein "
            "persönliches Gespräch zur Verfügung, um meine Qualifikationen und Motivation weiter zu erläutern. "
            ""
            "Mit freundlichen Grüßen,"
            "[Ihr Name]",]
    fool_detector_text(input_text)