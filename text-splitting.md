## Text splitting
In RAG, splitting text into chunks is key to making retrieval effective. But there’s more than one way to do it. Some methods are quick and dirty; others dig deeper to keep the meaning intact.

## Why Splitting
Direct model inference on long texts has some clear limitations:

<b>Finite Context Size:</b> Models have a maximum input size.
<b>Accuracy Decline with Size:</b> Even long context models can struggle when the input is too lengthy.
<b>Black-Box Retrieval:</b> Direct inference makes it harder to control what’s "retrieved"; you’re relying on the model’s judgment.
<b>Cost:</b> Processing extensive texts repeatedly can get expensive, especially if multiple questions need the same context.
