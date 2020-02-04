import React, { useState, useCallback } from "react";
import { Emotion } from "./predictions_interfaces";
import { Form, TextArea, Button, List, Icon } from "semantic-ui-react";

export const Predictions: React.FC = () => {
	const [text, setText] = useState("");
	const [predText, setPredText] = useState("");
	const [isSending, setIsSending] = useState(false);
	const [isUpdating, setIsUpdating] = useState(false);
	const [predictions, setPredictions] = useState([]);

	const getPredictions = useCallback(async () => {
		if (isSending) return;

		setIsSending(true);

		const textToSend: string = text;

		if (textToSend !== "") {
			const response: Response = await fetch("/", {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify(text)
			});

			if (response.ok) {
				response.json().then((preds: any[]) => {
					setPredictions(preds);
				});
				setPredText(text);
			}

			setText("");
			setIsSending(false);
		}
	}, [isSending, text]);

	const updatePrediction = useCallback(
		async (id: string, correct: boolean) => {
			if (isUpdating) return;

			setIsUpdating(true);

			let updated_preds = {
				id: id,
				text: predText,
				correct: correct
			};

			const response: Response = await fetch("/update_preds", {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify(updated_preds)
			});

			if (response.ok) {
				response.json().then(r => {
					console.log(r);
				});
			}

			setIsUpdating(false);
		},
		[predText, isUpdating]
	);

	const handleCorrectPredictionClicked = (id: string) => {
		updatePrediction(id, true);
		setPredictions(predictions.filter(p => p[0] !== id));

		if (predictions.length <= 1) {
			setPredText("");
		}
	};

	const handleIncorrectPredictionClicked = (id: string) => {
		updatePrediction(id, false);
		setPredictions(predictions.filter(p => p[0] !== id));

		if (predictions.length <= 1) {
			setPredText("");
		}
	};

	const showLoadingIcon = () => {
		return <div>Loading...</div>;
	};

	const renderPredictions = () => {
		return (
			<>
				<div style={{ marginTop: 12, marginBottom: 12, fontFamily: "Arial" }}>
					<h5>{predText}</h5>
					{predictions.map((p: any[]) => {
						let label: string = "";
						switch (p[0]) {
							case Emotion.ANGER:
								label = "Anger";
								break;
							case Emotion.ANTICIPATION:
								label = "Anticipation";
								break;
							case Emotion.DISGUST:
								label = "Disgust";
								break;
							case Emotion.FEAR:
								label = "Fear";
								break;
							case Emotion.JOY:
								label = "Joy";
								break;
							case Emotion.LOVE:
								label = "Love";
								break;
							case Emotion.OPTIMISM:
								label = "Optimism";
								break;
							case Emotion.PESSIMISM:
								label = "Pessimism";
								break;
							case Emotion.SADNESS:
								label = "Sadness";
								break;
							case Emotion.SURPRISE:
								label = "Surprise";
								break;
							case Emotion.TRUST:
								label = "Trust";
								break;
						}
						let content = label + " --> " + Math.round(p[1] * 100 + Number.EPSILON) / 100;
						return (
							<>
								<List divided verticalAlign='middle'>
									<List.Item>
										<List.Content floated='right'>
											<Button
												basic
												color='green'
												animated
												onClick={e => {
													handleCorrectPredictionClicked(p[0]);
												}}>
												<Button.Content visible>Correct</Button.Content>
												<Button.Content hidden>
													<Icon name='check' />
												</Button.Content>
											</Button>
											<Button
												basic
												color='red'
												animated
												onClick={e => {
													handleIncorrectPredictionClicked(p[0]);
												}}>
												<Button.Content visible>Incorrect</Button.Content>
												<Button.Content hidden>
													<Icon name='delete' />
												</Button.Content>
											</Button>
										</List.Content>
										<List.Content>{content}</List.Content>
									</List.Item>
								</List>
							</>
						);
					})}
				</div>
			</>
		);
	};

	return (
		<>
			<Form>
				<Form.Field>
					<TextArea
						value={text}
						onChange={e => {
							setText(e.currentTarget.value);
						}}></TextArea>
				</Form.Field>
				<Form.Field>
					<Button color={"twitter"} disabled={isSending} onClick={async () => getPredictions()}>
						Submit
					</Button>
				</Form.Field>
			</Form>
			{isUpdating ? showLoadingIcon() : renderPredictions()}
		</>
	);
};
