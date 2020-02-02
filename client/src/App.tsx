import React from "react";
import "./App.css";
import { Emotions } from "./components/emotions_component";
import { Emojis } from "./components/emotions_interfaces";
import { Container } from "semantic-ui-react";

const App: React.FC = () => {
	return (
		<>
			<Container style={{ marginTop: 20 }}>
				<div style={{ marginBottom: 20 }}>
					<h1>This AI has been trained to understand emotion from text.</h1>
					<div>
						<h4>Emotions are categorized into 11 distinct labels:</h4>
						<p style={{ fontSize: 13 }}>
							<strong>
								Anger {Emojis.ANGER} -- Anticipation {Emojis.ANTICIPATION} -- Disgust{" "}
								{Emojis.DISGUST} -- Fear {Emojis.FEAR} -- Joy {Emojis.JOY} -- Love {Emojis.LOVE} --
								Optimism {Emojis.OPTIMISM} -- Pessimism {Emojis.PESSIMISM} -- Sadness --{" "}
								{Emojis.SADNESS} -- Surprise {Emojis.SURPRISE} -- Trust {Emojis.TRUST}
							</strong>
						</p>
						<p style={{ fontSize: 13 }}>
							Teach the AI by labelling incorrect predictions with the emotion label(s) you think
							best represent the text.
						</p>
					</div>
				</div>
				<div>
					<Emotions />
				</div>
			</Container>
		</>
	);
};

export default App;
