import os

def get_rank(scores):
    # 从小到大排序，返回每个元素的原始索引在排序后的位置（排名）
    sorted_scores = sorted(scores)
    rank_dict = {score: i+1 for i, score in enumerate(sorted_scores)}
    ranks = [rank_dict[score] for score in scores]
    return ranks

def print_text_with_color_in_html(tokens, new_token_cnt, scores, show_token_border, path = 'tmp/colored_tokens.html'):
  scores = [-score for score in scores]
  color_grades = grade_score_list(scores)
  if new_token_cnt == 0:
      raw_tokens = tokens
      new_tokens = []
      raw_tokens_grades = color_grades
      new_tokens_grades = []
  else:
      raw_tokens = tokens[:-new_token_cnt]
      new_tokens = tokens[-new_token_cnt:]
      raw_tokens_grades = color_grades[:-new_token_cnt]
      new_tokens_grades = color_grades[-new_token_cnt:]
  if not os.path.exists(path):
      html_content = '<html><body>'
  else:
      with open(path, 'r', encoding='utf-8') as f:
          html_content = f.read()
          html_content = html_content.replace('</body></html>', '')
  html_content += '<p>'
  def print_to_html(tokens, tokens_grades, html_content, show_token_border):
      for idx, (token, grade) in enumerate(zip(tokens, tokens_grades)):
          if show_token_border and idx != 0:
              html_content += '|'
          color = get_color(grade)
          html_content += f"<span style='color: {color};'>{token}</span>"
      return html_content
  html_content = print_to_html(raw_tokens, raw_tokens_grades, html_content, show_token_border)
  if new_tokens:
      html_content += '->'
      html_content = print_to_html(new_tokens, new_tokens_grades, html_content, show_token_border)
  html_content += '</p>'
  html_content += '</body></html>'
  with open(path, 'w', encoding='utf-8') as f:
      f.write(html_content)

def grade_score_list(scores):
  ranks = get_rank(scores)
  grades = []
  cnt = len(scores)
  for rank in ranks:
      if rank <= 0.2 * cnt:
          grades.append(0)
      elif rank < int(0.8 * cnt):
          grades.append(1)
      else:
          grades.append(2)
  return grades

def get_color(grade):
  if grade == 0:
      return 'rgb(0, 255, 0)'
  elif grade == 1:
      # return 'rgb(153, 255, 153)'
      return 'rgb(122, 124, 243)' # 淡蓝色
  elif grade == 2:
      return 'rgb(255, 255, 255))'


# print_text_with_color_in_html(tokens, 0, attention_distance_scores, show_token_border=True)

