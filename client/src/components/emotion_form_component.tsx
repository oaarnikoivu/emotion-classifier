import React, { useState } from "react";
import { Form, Button, TextArea } from "semantic-ui-react";
import { EmotionProps, Emotions } from "./emotions_interfaces";

export const EmotionForm: React.FC<EmotionProps> = (props: EmotionProps) => {
	const [text, setText] = useState("");

	return (
		<Form>
			<Form.Field>
				<TextArea value={text} onChange={e => setText(e.currentTarget.value)}></TextArea>
			</Form.Field>
			<Form.Field>
				<Button
					onClick={async () => {
						const textToSend: string = text;
						const response: Response = await fetch("/", {
							method: "POST",
							headers: {
								"Content-Type": "application/json"
							},
							body: JSON.stringify(textToSend)
						});

						if (response.ok) {
							// retrieve prediction
							response.json().then(prediction => {
								switch (prediction) {
									case Emotions.ANGER:
										props.onNewText("Anger");
										break;
									case Emotions.ANTICIPATION:
										props.onNewText("Anticipation");
										break;
									case Emotions.DISGUST:
										props.onNewText("Disgust");
										break;
									case Emotions.FEAR:
										props.onNewText("Fear");
										break;
									case Emotions.JOY:
										props.onNewText("Joy");
										break;
									case Emotions.LOVE:
										props.onNewText("Love");
										break;
									case Emotions.OPTIMISM:
										props.onNewText("Optimism");
										break;
									case Emotions.PESSIMISM:
										props.onNewText("Pessimism");
										break;
									case Emotions.SADNESS:
										props.onNewText("Sadness");
										break;
									case Emotions.SURPRISE:
										props.onNewText("Surprise");
										break;
									case Emotions.TRUST:
										props.onNewText("Trust");
										break;
								}
							});

							setText("");
						}
					}}>
					Submit
				</Button>
			</Form.Field>
		</Form>
	);
};
