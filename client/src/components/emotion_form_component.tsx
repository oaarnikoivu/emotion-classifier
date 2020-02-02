import React, { useState } from "react";
import { Form, Button, TextArea } from "semantic-ui-react";
import { EmotionProps } from "./emotions_interfaces";

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

						if (textToSend !== "") {
							const response: Response = await fetch("/", {
								method: "POST",
								headers: {
									"Content-Type": "application/json"
								},
								body: JSON.stringify(textToSend)
							});

							if (response.ok) {
								// retrieve prediction
								response.json().then(predictions => {
									let this_arr = [];

									for (let i = 0; i < predictions.length; i++) {
										this_arr.push({
											id: predictions[i][0],
											value: predictions[i][1]
										});
									}
									props.onNewText(this_arr);
								});

								setText("");
							}
						}
					}}>
					Submit
				</Button>
			</Form.Field>
		</Form>
	);
};
