import React, { useState } from "react";
import { Form, Input, Button } from "semantic-ui-react";
import { EmotionProps } from "./emotions_interfaces";

export const EmotionForm: React.FC<EmotionProps> = (props: EmotionProps) => {
	const [text, setText] = useState("");

	return (
		<Form>
			<Form.Field>
				<Input placeholder='Text' value={text} onChange={e => setText(e.target.value)} />
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
							console.log("Response worked!");

							response.json().then(preds => {
								console.log(preds);
							});

							props.onNewText(text);
							setText("");
						}
					}}>
					Submit
				</Button>
			</Form.Field>
		</Form>
	);
};
