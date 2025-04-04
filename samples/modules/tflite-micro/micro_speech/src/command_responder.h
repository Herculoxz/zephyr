#ifndef COMMAND_RESPONDER_H_
#define COMMAND_RESPONDER_H_

// Called to handle the inference results (scores for silence, unknown, yes, no)
void RespondToCommand(float silence_score, float unknown_score, float yes_score, float no_score);

#endif  // COMMAND_RESPONDER_H_