import React from "react";
import { EmojiProps } from "../emoji/emoji_interfaces";

export const Emoji: React.FC<EmojiProps> = (props: EmojiProps) => {
	return (
		<div style={{ marginTop: 20 }}>
			<span
				style={{ fontSize: 40 }}
				className='emoji'
				role='img'
				aria-label={props.label ? props.label : ""}
				aria-hidden={props.label ? "false" : "true"}>
				{props.symbol}
			</span>
		</div>
	);
};
