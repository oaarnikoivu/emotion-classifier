import React from "react";
import { EmojiProps } from "../emoji/emoji_interfaces";

export const Emoji: React.FC<EmojiProps> = (props: EmojiProps) => {
	return (
		<div style={{ marginTop: 4 }}>
			{props.showLabel ? props.label : undefined}
			<span
				style={{ fontSize: props.fontSize ? props.fontSize : 20, marginLeft: 4 }}
				className='emoji'
				role='img'
				aria-label={props.label ? props.label : ""}
				aria-hidden={props.label ? "false" : "true"}>
				{props.symbol}
			</span>
		</div>
	);
};
