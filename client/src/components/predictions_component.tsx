import React, { useState, useCallback } from "react";
import { Emotion } from "./predictions_interfaces";
import { Form, TextArea, Button, List, Icon } from "semantic-ui-react";
import { Colors } from "../utils/colors";

const API_PREDICTION_ENDPOINT: string = "https://server.ainoa.wtf/predictions";
const API_UPDATE_PREDICTION_ENDPOINT: string = "https://server.ainoa.wtf/update_preds";

export const Predictions: React.FC = () => {
	const [text, setText] = useState("");
	const [predText, setPredText] = useState("");
	const [isSending, setIsSending] = useState(false);
	const [isUpdating, setIsUpdating] = useState(false);
	const [predictions, setPredictions] = useState([]);
	const [attnWeights, setAttnWeights] = useState([]);

	const getPredictions = useCallback(async () => {
		if (isSending) return;

		setIsSending(true);

		const textToSend: string = text;
		const textLength = textToSend.length;

		if (textToSend !== "") {
			const response: Response = await fetch(API_PREDICTION_ENDPOINT, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify([text, textLength]),
			});

			if (response.ok) {
				response.json().then((r: any[]) => {
					setPredictions(r[0]);
					setAttnWeights(r[1]);
				});
				setPredText(text.replace(/[^A-Za-z0-9\s]/g, "").replace(/\s{2,}/g, " "));
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
				correct: correct,
			};

			const response: Response = await fetch(API_UPDATE_PREDICTION_ENDPOINT, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(updated_preds),
			});

			if (response.ok) {
				response.json().then((r) => {
					console.log(r);
				});
			}

			setIsUpdating(false);
		},
		[predText, isUpdating]
	);

	const handleCorrectPredictionClicked = (id: string) => {
		updatePrediction(id, true);
		setPredictions(predictions.filter((p) => p[0] !== id));

		if (predictions.length <= 1) {
			setPredText("");
		}
	};

	const handleIncorrectPredictionClicked = (id: string) => {
		updatePrediction(id, false);
		setPredictions(predictions.filter((p) => p[0] !== id));

		if (predictions.length <= 1) {
			setPredText("");
		}
	};

	const showLoadingIcon = () => {
		return <div>Loading...</div>;
	};

	const applyColorsToWeights = (searchWords: string[], text: string): any => {
		let fontSize: number;
		let color: string;
		if (searchWords.includes(text)) {
			if (text === searchWords[0]) {
				color = Colors.DARKEST;
				fontSize = 20;
			} else if (text === searchWords[1]) {
				color = Colors.DARK;
				fontSize = 18;
			} else if (text === searchWords[2]) {
				color = Colors.INBETWEEN;
				fontSize = 16;
			} else if (text === searchWords[3]) {
				color = Colors.LIGHTER;
				fontSize = 14;
			} else if (text === searchWords[4]) {
				color = Colors.LIGHT;
				fontSize = 13;
			} else {
				color = Colors.LIGHTEST;
				fontSize = 13;
			}
		}
		return [fontSize, color];
	};

	const renderPredictions = () => {
		let searchWords: string[] = [];
		attnWeights.forEach((weight) => {
			searchWords.push(weight[0]);
		});

		let text: any;

		if (predText) {
			text = predText.split(" ").map(function (a, i) {
				let [first, second] = applyColorsToWeights(searchWords, a);
				let fontSize = first;
				let color = second;

				return (
					<>
						<div style={{ display: "flex", marginBottom: 12 }}>
							<span
								style={{
									marginRight: 9,
									color: color,
									fontSize: fontSize,
									borderRadius: 2,
									padding: 4,
								}}
								key={i}>
								{a}
							</span>
						</div>
					</>
				);
			});
		}

		return (
			<>
				<div style={{ marginTop: 36, marginBottom: 12, fontFamily: "Arial" }}>
					<div style={{ display: "flex", flexWrap: "wrap" }}>{text}</div>
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
												onClick={(e) => {
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
												onClick={(e) => {
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
						onChange={(e) => {
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
