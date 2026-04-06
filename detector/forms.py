from __future__ import annotations

from django import forms


class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(
        help_text="Upload a CSV file with text and label columns."
    )


class PredictionForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 5}),
        label="Text to analyze",
        help_text="Enter one sentence per line to get multiple predictions.",
    )
