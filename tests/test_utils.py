from climanet.utils import setup_logging
from tbparse import SummaryReader


def test_setup_logging(tmp_path):
    writer = setup_logging(tmp_path)
    log_text = "This is a test log entry."
    writer.add_scalar("Test Scalar", 42)
    writer.add_text("Test Text", log_text)
    writer.close()

    # Test that there is one event file
    # The file should have "UTC" keyword in timestamp suffix
    assert len(list(tmp_path.glob("events*UTC*"))) == 1

    # Load the events file with SummaryReader
    reader = SummaryReader(tmp_path)
    assert reader.text["value"].iloc[0] == log_text
