from climanet.utils import setup_logging


def test_setup_logging(tmp_path):
    writer = setup_logging(str(tmp_path))
    writer.add_text("test", "This is a test log entry.")
    writer.close()

    # Test that there is at least one events file
    # The filename should have "UTC" keyword, which is the timestamp suffix
    assert any(tmp_path.glob("events.out.*UTC*"))
