from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

import pytest

from airaola.notifications.email_delivery import (
    DEFAULT_SMTP_HOST,
    DEFAULT_SMTP_PORT,
    EmailConfiguration,
    attach_report_files,
    build_report_email,
    load_email_configuration,
    send_report_email,
)
from airaola.reporting.weekly_report import (
    WeeklyReport,
)


def _configuration() -> EmailConfiguration:
    """Build synthetic SMTP configuration."""

    return EmailConfiguration(
        sender_email="manager@example.com",
        recipient_email="owner@example.com",
        app_password="abcdefghijklmnop",
    )


def _report(
    text_path: Path | None = None,
    html_path: Path | None = None,
) -> WeeklyReport:
    """Build a synthetic weekly decision report."""

    return WeeklyReport(
        gameweek=1,
        text_content=(
            "Project Airaola\n"
            "Gameweek 1 Decision Report\n"
            "Captain: Test Captain\n"
        ),
        html_content=(
            "<!DOCTYPE html>"
            "<html>"
            "<body>"
            "<h1>Project Airaola</h1>"
            "<p>Gameweek 1 Decision Report</p>"
            "</body>"
            "</html>"
        ),
        text_path=text_path,
        html_path=html_path,
    )


def test_loads_required_email_configuration() -> None:
    environment = {
        "AIRAOLA_EMAIL_SENDER": "manager@example.com",
        "AIRAOLA_EMAIL_RECIPIENT": "owner@example.com",
        "AIRAOLA_EMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
    }

    configuration = load_email_configuration(
        environment
    )

    assert (
        configuration.sender_email
        == "manager@example.com"
    )

    assert (
        configuration.recipient_email
        == "owner@example.com"
    )

    assert (
        configuration.app_password
        == "abcdefghijklmnop"
    )

    assert (
        configuration.smtp_host
        == DEFAULT_SMTP_HOST
    )

    assert (
        configuration.smtp_port
        == DEFAULT_SMTP_PORT
    )


def test_loads_custom_smtp_configuration() -> None:
    environment = {
        "AIRAOLA_EMAIL_SENDER": "manager@example.com",
        "AIRAOLA_EMAIL_RECIPIENT": "owner@example.com",
        "AIRAOLA_EMAIL_APP_PASSWORD": "abcdefghijklmnop",
        "AIRAOLA_EMAIL_SMTP_HOST": "smtp.example.com",
        "AIRAOLA_EMAIL_SMTP_PORT": "2525",
    }

    configuration = load_email_configuration(
        environment
    )

    assert (
        configuration.smtp_host
        == "smtp.example.com"
    )

    assert configuration.smtp_port == 2525


def test_missing_required_configuration_is_rejected() -> None:
    environment = {
        "AIRAOLA_EMAIL_SENDER": "manager@example.com",
    }

    with pytest.raises(
        ValueError,
        match="missing environment variables",
    ):
        load_email_configuration(
            environment
        )


def test_invalid_sender_email_is_rejected() -> None:
    environment = {
        "AIRAOLA_EMAIL_SENDER": "invalid-address",
        "AIRAOLA_EMAIL_RECIPIENT": "owner@example.com",
        "AIRAOLA_EMAIL_APP_PASSWORD": "abcdefghijklmnop",
    }

    with pytest.raises(
        ValueError,
        match="must be a valid email address",
    ):
        load_email_configuration(
            environment
        )


def test_invalid_smtp_port_is_rejected() -> None:
    environment = {
        "AIRAOLA_EMAIL_SENDER": "manager@example.com",
        "AIRAOLA_EMAIL_RECIPIENT": "owner@example.com",
        "AIRAOLA_EMAIL_APP_PASSWORD": "abcdefghijklmnop",
        "AIRAOLA_EMAIL_SMTP_PORT": "invalid",
    }

    with pytest.raises(
        ValueError,
        match="SMTP port must be an integer",
    ):
        load_email_configuration(
            environment
        )


def test_builds_multipart_report_email() -> None:
    message = build_report_email(
        report=_report(),
        configuration=_configuration(),
    )

    assert isinstance(
        message,
        EmailMessage,
    )

    assert (
        message["Subject"]
        == "Project Airaola | Gameweek 1 Decision Report"
    )

    assert (
        message["From"]
        == "manager@example.com"
    )

    assert (
        message["To"]
        == "owner@example.com"
    )

    assert message.is_multipart()

    body = message.get_body(
        preferencelist=(
            "plain",
        )
    )

    html = message.get_body(
        preferencelist=(
            "html",
        )
    )

    assert body is not None
    assert html is not None

    assert (
        "Captain: Test Captain"
        in body.get_content()
    )

    assert (
        "<h1>Project Airaola</h1>"
        in html.get_content()
    )


def test_empty_text_content_is_rejected() -> None:
    report = WeeklyReport(
        gameweek=1,
        text_content="",
        html_content="<html></html>",
    )

    with pytest.raises(
        ValueError,
        match="text content cannot be empty",
    ):
        build_report_email(
            report=report,
            configuration=_configuration(),
        )


def test_empty_html_content_is_rejected() -> None:
    report = WeeklyReport(
        gameweek=1,
        text_content="Report",
        html_content="",
    )

    with pytest.raises(
        ValueError,
        match="HTML content cannot be empty",
    ):
        build_report_email(
            report=report,
            configuration=_configuration(),
        )


def test_attaches_saved_report_files(
    tmp_path: Path,
) -> None:
    text_path = (
        tmp_path
        / "gameweek_1_report.txt"
    )

    html_path = (
        tmp_path
        / "gameweek_1_report.html"
    )

    text_path.write_text(
        "Text report",
        encoding="utf-8",
    )

    html_path.write_text(
        "<html>HTML report</html>",
        encoding="utf-8",
    )

    report = _report(
        text_path=text_path,
        html_path=html_path,
    )

    message = build_report_email(
        report=report,
        configuration=_configuration(),
    )

    attach_report_files(
        message=message,
        report=report,
    )

    filenames = {
        attachment.get_filename()
        for attachment in message.iter_attachments()
    }

    assert filenames == {
        "gameweek_1_report.txt",
        "gameweek_1_report.html",
    }


def test_missing_attachment_is_rejected(
    tmp_path: Path,
) -> None:
    report = _report(
        text_path=(
            tmp_path
            / "missing_report.txt"
        ),
    )

    message = build_report_email(
        report=report,
        configuration=_configuration(),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Report attachment not found",
    ):
        attach_report_files(
            message=message,
            report=report,
        )


def test_successful_email_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeSMTP:
        """Simulate a successful SMTP session."""

        def __init__(
            self,
            host: str,
            port: int,
            timeout: float,
        ) -> None:
            assert host == DEFAULT_SMTP_HOST
            assert port == DEFAULT_SMTP_PORT
            assert timeout == 30.0

            calls.append(
                "connect"
            )

        def __enter__(self):
            calls.append(
                "enter"
            )

            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ) -> None:
            calls.append(
                "exit"
            )

        def ehlo(self) -> None:
            calls.append(
                "ehlo"
            )

        def starttls(
            self,
            context,
        ) -> None:
            assert context is not None

            calls.append(
                "starttls"
            )

        def login(
            self,
            sender_email: str,
            app_password: str,
        ) -> None:
            assert (
                sender_email
                == "manager@example.com"
            )

            assert (
                app_password
                == "abcdefghijklmnop"
            )

            calls.append(
                "login"
            )

        def send_message(
            self,
            message: EmailMessage,
        ) -> dict:
            assert (
                message["To"]
                == "owner@example.com"
            )

            calls.append(
                "send"
            )

            return {}

    monkeypatch.setattr(
        "airaola.notifications.email_delivery.smtplib.SMTP",
        FakeSMTP,
    )

    result = send_report_email(
        report=_report(),
        configuration=_configuration(),
        attach_files=False,
    )

    assert result.sent is True

    assert (
        result.recipient_email
        == "owner@example.com"
    )

    assert (
        result.subject
        == "Project Airaola | Gameweek 1 Decision Report"
    )

    assert result.error_message is None

    assert calls == [
        "connect",
        "enter",
        "ehlo",
        "starttls",
        "ehlo",
        "login",
        "send",
        "exit",
    ]


def test_smtp_failure_returns_unsent_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSMTP:
        """Simulate an SMTP connection failure."""

        def __init__(
            self,
            host: str,
            port: int,
            timeout: float,
        ) -> None:
            raise OSError(
                "Synthetic connection failure"
            )

    monkeypatch.setattr(
        "airaola.notifications.email_delivery.smtplib.SMTP",
        FailingSMTP,
    )

    result = send_report_email(
        report=_report(),
        configuration=_configuration(),
        attach_files=False,
    )

    assert result.sent is False

    assert (
        "Synthetic connection failure"
        in str(
            result.error_message
        )
    )


def test_refused_recipient_returns_unsent_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RefusingSMTP:
        """Simulate a recipient refusal."""

        def __init__(
            self,
            host: str,
            port: int,
            timeout: float,
        ) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ) -> None:
            pass

        def ehlo(self) -> None:
            pass

        def starttls(
            self,
            context,
        ) -> None:
            pass

        def login(
            self,
            sender_email: str,
            app_password: str,
        ) -> None:
            pass

        def send_message(
            self,
            message: EmailMessage,
        ) -> dict:
            return {
                "owner@example.com": (
                    550,
                    b"Mailbox unavailable",
                )
            }

    monkeypatch.setattr(
        "airaola.notifications.email_delivery.smtplib.SMTP",
        RefusingSMTP,
    )

    result = send_report_email(
        report=_report(),
        configuration=_configuration(),
        attach_files=False,
    )

    assert result.sent is False

    assert (
        "refused one or more recipients"
        in str(
            result.error_message
        )
    )