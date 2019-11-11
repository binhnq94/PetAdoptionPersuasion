import re


def _create_remove_pattern():
    urls = r"""             # Capture 1: entire matched URL
      (?:
      (ftp|http)s?:               # URL protocol and colon
        (?:
          /{1,3}            # 1-3 slashes
          |                 #   or
          [a-z0-9%]         # Single letter or digit or '%'
                            # (Trying not to match e.g. "URI::Escape")
        )
        |                   #   or
                            # looks like domain name followed by a slash:
        [a-z0-9.\-]+[.]
        (?:[a-z]{2,13})
        /
      )
      (?:                                  # One or more:
        [^\s()<>{}\[\]]+                   # Run of non-space, non-()<>{}[]
        |                                  #   or
        \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
        |
        \([^\s]+?\)                        # balanced parens, non-recursive: (...)
      )+
      (?:                                  # End with:
        \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
        |
        \([^\s]+?\)                        # balanced parens, non-recursive: (...)
        |                                  #   or
        [^\s`!()\[\]{};:'".,<>?«»“”‘’]     # not a space or one of these punct chars
      )
      |                        # OR, the following to match naked domains:
      (?:
        (?<!@)                 # not preceded by a @, avoid matching foo@_gmail.com_
        [a-z0-9]+
        (?:[.\-][a-z0-9]+)*
        [.]
        (?:[a-z]{2,13})
        \b
        /?
        (?!@)                  # not succeeded by a @,
                               # avoid matching "foo.na" in "foo.na@example.com"
      )
    """
    url = re.compile(urls)
    email = re.compile(r'[\w.!#$%&’*+\/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)*')


    # patterns = "(" + "|".join(patterns) + ")"
    # return re.compile(patterns, re.UNICODE)
    return {
        '__url__': url,
        '__email__': email
    }


REPLACE_EXPS = _create_remove_pattern()


def clean_text(text):
    for key in ['__url__', '__email__']:
        exp = REPLACE_EXPS[key]
        text = exp.sub(key, text)
    return text
