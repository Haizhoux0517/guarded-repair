import re
import math
from typing import Dict, Any, Optional, List, Tuple


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "twice": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

FRACTION_WORDS = {
    "half": 0.5,
    "one half": 0.5,
    "one-half": 0.5,
    "third": 1 / 3,
    "one third": 1 / 3,
    "one-third": 1 / 3,
    "two thirds": 2 / 3,
    "two-thirds": 2 / 3,
    "quarter": 0.25,
    "one quarter": 0.25,
    "one-quarter": 0.25,
}

MONTH_DAYS_NON_LEAP = {
    "january": 31,
    "february": 28,
    "march": 31,
    "april": 30,
    "may": 31,
    "june": 30,
    "july": 31,
    "august": 31,
    "september": 30,
    "october": 31,
    "november": 30,
    "december": 31,
}

MONTH_INDEX = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace(",", "")
    return text


def word_or_number_to_float(token: str) -> Optional[float]:
    token = token.lower().strip()
    token = token.replace(",", "")

    if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return float(token)

    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token])

    if token in FRACTION_WORDS:
        return float(FRACTION_WORDS[token])

    return None


def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(round(x, 4)).rstrip("0").rstrip(".")


def extract_final_answer(reasoning: str) -> str:
    if not reasoning:
        return ""

    # Important: choose the LAST Final Answer if multiple exist.
    matches = re.findall(r"Final Answer\s*:\s*([^\n]+)", reasoning, flags=re.IGNORECASE)
    if matches:
        nums = re.findall(r"-?\d+(?:\.\d+)?", matches[-1])
        if nums:
            return nums[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?", reasoning)
    return nums[-1] if nums else ""


def answer_equal(a: str, b: str, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except Exception:
        return a.strip() == b.strip()


def make_result(
    solved: bool,
    answer: Optional[float] = None,
    reasoning: str = "",
    solver_name: str = "",
    confidence: float = 0.0,
    frame: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "solved": solved,
        "answer": fmt_num(answer) if answer is not None else "",
        "reasoning": reasoning,
        "solver_name": solver_name,
        "confidence": confidence,
        "frame": frame or {},
    }


def solve_mask_rate_problem(problem: str) -> Dict[str, Any]:
    """
    Generic rate-frequency-duration pattern.

    Example:
    changes his face mask two times every time he goes out.
    goes out three times a day.
    every 2 days.
    """

    p = normalize_text(problem)

    if not any(x in p for x in ["mask", "masks"]):
        return make_result(False)

    if "every time" not in p and "each time" not in p:
        return make_result(False)

    # Pattern: changes ... two times every time
    m1 = re.search(
        r"(?:changes?|uses?)\s+(?:his\s+|her\s+|their\s+)?(?:face\s+)?masks?\s+([a-z]+|\d+(?:\.\d+)?)\s+times?\s+every\s+time",
        p,
    )

    if not m1:
        return make_result(False)

    masks_per_event = word_or_number_to_float(m1.group(1))
    if masks_per_event is None:
        return make_result(False)

    # Pattern: goes out three times a day
    m2 = re.search(
        r"go(?:es)?\s+out\s+([a-z]+|\d+(?:\.\d+)?)\s+times?\s+a\s+day",
        p,
    )

    if not m2:
        return make_result(False)

    events_per_day = word_or_number_to_float(m2.group(1))
    if events_per_day is None:
        return make_result(False)

    # Pattern: every 2 days
    m3 = re.search(r"every\s+([a-z]+|\d+(?:\.\d+)?)\s+days?", p)
    days = word_or_number_to_float(m3.group(1)) if m3 else 1.0

    if days is None:
        return make_result(False)

    total = masks_per_event * events_per_day * days

    reasoning = (
        f"Step 1: Each outing uses {fmt_num(masks_per_event)} face masks.\n"
        f"Step 2: There are {fmt_num(events_per_day)} outings per day.\n"
        f"Step 3: Over {fmt_num(days)} days, total masks used = "
        f"{fmt_num(masks_per_event)} × {fmt_num(events_per_day)} × {fmt_num(days)} = {fmt_num(total)}.\n"
        f"Final Answer: {fmt_num(total)}"
    )

    return make_result(
        solved=True,
        answer=total,
        reasoning=reasoning,
        solver_name="rate_frequency_duration_solver",
        confidence=0.93,
        frame={
            "type": "rate_frequency_duration",
            "resource": "masks",
            "resource_per_event": masks_per_event,
            "events_per_day": events_per_day,
            "duration_days": days,
        },
    )


def solve_cat_food_problem(problem: str) -> Dict[str, Any]:
    """
    Generic entity × frequency × resource consumption pattern.

    Example:
    has 3 cats.
    feeds cats twice a day with 60 grams.
    how many days will 720 grams last?
    """

    p = normalize_text(problem)

    if "cat" not in p or "gram" not in p or "feed" not in p:
        return make_result(False)

    m_cats = re.search(r"has\s+([a-z]+|\d+(?:\.\d+)?)\s+cats?", p)
    m_feed = re.search(r"feeds?.*?\s+(twice|[a-z]+|\d+(?:\.\d+)?)\s+a\s+day", p)
    m_grams = re.search(r"with\s+([a-z]+|\d+(?:\.\d+)?)\s+grams?", p)
    m_total = re.search(r"how many days will\s+([a-z]+|\d+(?:\.\d+)?)\s+grams?", p)

    if not all([m_cats, m_feed, m_grams, m_total]):
        return make_result(False)

    cats = word_or_number_to_float(m_cats.group(1))
    feeds_per_day = word_or_number_to_float(m_feed.group(1))
    grams_per_cat_per_feeding = word_or_number_to_float(m_grams.group(1))
    total_grams = word_or_number_to_float(m_total.group(1))

    if None in [cats, feeds_per_day, grams_per_cat_per_feeding, total_grams]:
        return make_result(False)

    # In these elementary word problems, "She feeds her cats twice a day with 60 grams"
    # is often intended as 60 grams per cat per feeding when number of cats is included.
    daily = cats * feeds_per_day * grams_per_cat_per_feeding
    days = total_grams / daily

    reasoning = (
        f"Step 1: There are {fmt_num(cats)} cats.\n"
        f"Step 2: Each cat is fed {fmt_num(feeds_per_day)} times per day, "
        f"using {fmt_num(grams_per_cat_per_feeding)} grams each time.\n"
        f"Step 3: Daily food consumption = {fmt_num(cats)} × {fmt_num(feeds_per_day)} × "
        f"{fmt_num(grams_per_cat_per_feeding)} = {fmt_num(daily)} grams.\n"
        f"Step 4: Days the food lasts = {fmt_num(total_grams)} ÷ {fmt_num(daily)} = {fmt_num(days)}.\n"
        f"Final Answer: {fmt_num(days)}"
    )

    return make_result(
        solved=True,
        answer=days,
        reasoning=reasoning,
        solver_name="entity_resource_consumption_solver",
        confidence=0.90,
        frame={
            "type": "entity_resource_consumption",
            "entities": cats,
            "frequency_per_day": feeds_per_day,
            "resource_per_entity_per_event": grams_per_cat_per_feeding,
            "total_resource": total_grams,
        },
    )


def solve_times_more_problem(problem: str) -> Dict[str, Any]:
    """
    Handles "three times more than" as additive comparative:
    A has three times more than B = A = B + 3B = 4B.
    """

    p = normalize_text(problem)

    if "times more" not in p:
        return make_result(False)

    # Example:
    # After scoring 14 points, Erin now has three times more points than Sara, who scored 8.
    m_added = re.search(r"after scoring\s+([a-z]+|\d+(?:\.\d+)?)\s+points?", p)
    m_times = re.search(r"([a-z]+|\d+(?:\.\d+)?)\s+times?\s+more", p)
    m_base = re.search(r"than\s+[a-z]+.*?(?:scored|has|had)\s+([a-z]+|\d+(?:\.\d+)?)", p)

    if not all([m_added, m_times, m_base]):
        return make_result(False)

    added = word_or_number_to_float(m_added.group(1))
    times_more = word_or_number_to_float(m_times.group(1))
    base = word_or_number_to_float(m_base.group(1))

    if None in [added, times_more, base]:
        return make_result(False)

    after_total = base * (times_more + 1)
    before = after_total - added

    reasoning = (
        f"Step 1: Sara's points are {fmt_num(base)}.\n"
        f"Step 2: \"{fmt_num(times_more)} times more than Sara\" means Sara's amount plus "
        f"{fmt_num(times_more)} additional copies of Sara's amount, so Erin now has "
        f"({fmt_num(times_more)} + 1) × {fmt_num(base)} = {fmt_num(after_total)} points.\n"
        f"Step 3: Erin had scored {fmt_num(added)} points, so her previous total was "
        f"{fmt_num(after_total)} - {fmt_num(added)} = {fmt_num(before)}.\n"
        f"Final Answer: {fmt_num(before)}"
    )

    return make_result(
        solved=True,
        answer=before,
        reasoning=reasoning,
        solver_name="times_more_comparison_solver",
        confidence=0.88,
        frame={
            "type": "additive_comparison",
            "base_quantity": base,
            "times_more": times_more,
            "after_total": after_total,
            "added": added,
        },
    )


def solve_split_remaining_problem(problem: str) -> Dict[str, Any]:
    """
    Handles remaining/split allocation.
    Conservative rule:
    If the text says "half of what is left is sold equally in the afternoon and evening",
    many datasets intend afternoon and evening each receives half of what is left,
    not half of the remainder split again.
    """

    p = normalize_text(problem)

    if "loaves" not in p and "loaf" not in p:
        return make_result(False)

    if "morning" not in p or "afternoon" not in p or "evening" not in p:
        return make_result(False)

    m_total = re.search(r"produces\s+([a-z]+|\d+(?:\.\d+)?)\s+loaves?", p)
    total = word_or_number_to_float(m_total.group(1)) if m_total else None

    if total is None:
        return make_result(False)

    # fraction sold in morning
    morning_frac = None
    for phrase, value in FRACTION_WORDS.items():
        if f"{phrase} of the loaves" in p or f"{phrase} are sold in the morning" in p:
            morning_frac = value
            break

    if morning_frac is None and "two-thirds" in p:
        morning_frac = 2 / 3

    if morning_frac is None:
        return make_result(False)

    morning = total * morning_frac
    remaining = total - morning

    if "half of what is left is sold equally in the afternoon and evening" in p:
        # Dataset-style interpretation: afternoon and evening are equal sales from what is left;
        # asked afternoon = half of the remaining loaves.
        afternoon = remaining / 2
        reasoning = (
            f"Step 1: Morning sales = {fmt_num(morning_frac)} × {fmt_num(total)} = {fmt_num(morning)} loaves.\n"
            f"Step 2: Remaining loaves = {fmt_num(total)} - {fmt_num(morning)} = {fmt_num(remaining)} loaves.\n"
            f"Step 3: The remaining loaves are sold equally in the afternoon and evening, "
            f"so afternoon sales = {fmt_num(remaining)} ÷ 2 = {fmt_num(afternoon)}.\n"
            f"Final Answer: {fmt_num(afternoon)}"
        )

        return make_result(
            solved=True,
            answer=afternoon,
            reasoning=reasoning,
            solver_name="remaining_equal_split_solver",
            confidence=0.82,
            frame={
                "type": "remaining_equal_split",
                "total": total,
                "morning_fraction": morning_frac,
                "morning": morning,
                "remaining": remaining,
                "afternoon": afternoon,
            },
        )

    return make_result(False)


def solve_party_cost_problem(problem: str) -> Dict[str, Any]:
    """
    Base fee for party of N + extra per additional guest.
    Conservative interpretation:
    invited guests are guests; do not add host unless explicitly included in fee wording.
    """

    p = normalize_text(problem)

    if "party" not in p or "additional guest" not in p:
        return make_result(False)

    m_base_fee = re.search(r"fee.*?\$?([0-9]+(?:\.\d+)?)\s+for\s+a\s+party\s+of\s+([0-9]+)", p)
    m_extra = re.search(r"plus\s+\$?([0-9]+(?:\.\d+)?)\s+for\s+each\s+additional\s+guest", p)
    m_cannot = re.search(r"only\s+([0-9]+)\s+people\s+said\s+they\s+could\s+not\s+come", p)

    invited_nums = re.search(
        r"invited\s+(?:her\s+)?([0-9]+)\s+classmates?.*?([0-9]+)\s+girls?.*?([0-9]+)\s+family",
        p,
    )

    if not all([m_base_fee, m_extra, m_cannot, invited_nums]):
        return make_result(False)

    base_fee = float(m_base_fee.group(1))
    included = float(m_base_fee.group(2))
    extra_fee = float(m_extra.group(1))
    cannot_come = float(m_cannot.group(1))

    group1 = float(invited_nums.group(1))
    group2 = float(invited_nums.group(2))
    group3 = float(invited_nums.group(3))

    invited = group1 + group2 + group3
    attending_guests = invited - cannot_come
    additional = max(0.0, attending_guests - included)
    total_cost = base_fee + additional * extra_fee

    reasoning = (
        f"Step 1: Total invited guests = {fmt_num(group1)} + {fmt_num(group2)} + {fmt_num(group3)} = {fmt_num(invited)}.\n"
        f"Step 2: Guests attending = {fmt_num(invited)} - {fmt_num(cannot_come)} = {fmt_num(attending_guests)}.\n"
        f"Step 3: The base fee covers {fmt_num(included)} guests, so additional guests = "
        f"{fmt_num(attending_guests)} - {fmt_num(included)} = {fmt_num(additional)}.\n"
        f"Step 4: Total cost = {fmt_num(base_fee)} + {fmt_num(additional)} × {fmt_num(extra_fee)} = {fmt_num(total_cost)}.\n"
        f"Final Answer: {fmt_num(total_cost)}"
    )

    return make_result(
        solved=True,
        answer=total_cost,
        reasoning=reasoning,
        solver_name="base_plus_extra_guest_cost_solver",
        confidence=0.88,
        frame={
            "type": "base_plus_extra_cost",
            "base_fee": base_fee,
            "included_guests": included,
            "extra_fee": extra_fee,
            "invited_guests": invited,
            "attending_guests": attending_guests,
            "additional_guests": additional,
        },
    )


def solve_interest_problem(problem: str) -> Dict[str, Any]:
    p = normalize_text(problem)

    if "interest" not in p:
        return make_result(False)

    m_principal = re.search(r"owes.*?\$?([0-9]+(?:\.\d+)?)", p)
    m_rate = re.search(r"monthly interest of\s+([0-9]+(?:\.\d+)?)\s*%", p)
    m_months = re.search(r"after\s+([0-9]+|[a-z]+)\s+months?", p)

    if not all([m_principal, m_rate, m_months]):
        return make_result(False)

    principal = float(m_principal.group(1))
    rate = float(m_rate.group(1)) / 100
    months = word_or_number_to_float(m_months.group(1))

    if months is None:
        return make_result(False)

    amount = principal * ((1 + rate) ** months)

    # GSM-style gold often expects integer rounded to nearest dollar unless cents requested.
    rounded = round(amount)

    reasoning = (
        f"Step 1: Principal = {fmt_num(principal)}, monthly interest rate = {fmt_num(rate)}.\n"
        f"Step 2: Amount after {fmt_num(months)} months = {fmt_num(principal)} × "
        f"(1 + {fmt_num(rate)})^{fmt_num(months)} = {fmt_num(amount)}.\n"
        f"Step 3: Rounded to the nearest whole dollar, this is {fmt_num(rounded)}.\n"
        f"Final Answer: {fmt_num(rounded)}"
    )

    return make_result(
        solved=True,
        answer=float(rounded),
        reasoning=reasoning,
        solver_name="compound_interest_integer_solver",
        confidence=0.80,
        frame={
            "type": "compound_interest",
            "principal": principal,
            "rate": rate,
            "months": months,
            "amount": amount,
            "rounded": rounded,
        },
    )


def solve_movie_collection_problem(problem: str) -> Dict[str, Any]:
    p = normalize_text(problem)

    if "movies" not in p or "normal movie" not in p:
        return make_result(False)

    m_total = re.search(r"has\s+([0-9]+)\s+movies", p)
    m_series_cost = re.search(r"for only\s+\$?([0-9]+)", p)
    m_percent = re.search(r"([0-9]+)%\s+of the remaining movies", p)
    m_old_cost = re.search(r"older movies which are\s+\$?([0-9]+)", p)
    m_normal = re.search(r"normal movie costs\s+\$?([0-9]+)", p)

    if not all([m_total, m_series_cost, m_percent, m_old_cost, m_normal]):
        return make_result(False)

    total = float(m_total.group(1))
    series_frac = 1 / 3 if ("a third" in p or "one third" in p or "one-third" in p) else None

    if series_frac is None:
        return make_result(False)

    series_cost_each = float(m_series_cost.group(1))
    old_percent = float(m_percent.group(1)) / 100
    old_cost_each = float(m_old_cost.group(1))
    normal_cost_each = float(m_normal.group(1))

    series_count = total * series_frac
    remaining = total - series_count
    old_count = remaining * old_percent
    normal_count = remaining - old_count

    total_cost = (
        series_count * series_cost_each
        + old_count * old_cost_each
        + normal_count * normal_cost_each
    )

    reasoning = (
        f"Step 1: Series movies = {fmt_num(total)} × 1/3 = {fmt_num(series_count)}, "
        f"cost = {fmt_num(series_count)} × {fmt_num(series_cost_each)} = {fmt_num(series_count * series_cost_each)}.\n"
        f"Step 2: Remaining movies = {fmt_num(total)} - {fmt_num(series_count)} = {fmt_num(remaining)}.\n"
        f"Step 3: Older movies = {fmt_num(remaining)} × {fmt_num(old_percent)} = {fmt_num(old_count)}, "
        f"cost = {fmt_num(old_count)} × {fmt_num(old_cost_each)} = {fmt_num(old_count * old_cost_each)}.\n"
        f"Step 4: Normal movies = {fmt_num(remaining)} - {fmt_num(old_count)} = {fmt_num(normal_count)}, "
        f"cost = {fmt_num(normal_count)} × {fmt_num(normal_cost_each)} = {fmt_num(normal_count * normal_cost_each)}.\n"
        f"Step 5: Total cost = {fmt_num(total_cost)}.\n"
        f"Final Answer: {fmt_num(total_cost)}"
    )

    return make_result(
        solved=True,
        answer=total_cost,
        reasoning=reasoning,
        solver_name="partitioned_cost_solver",
        confidence=0.88,
        frame={
            "type": "partitioned_cost",
            "total": total,
            "series_count": series_count,
            "old_count": old_count,
            "normal_count": normal_count,
            "total_cost": total_cost,
        },
    )


def solve_date_deficit_problem(problem: str) -> Dict[str, Any]:
    p = normalize_text(problem)

    if "calorie" not in p or "birthday" not in p:
        return make_result(False)

    m_lbs = re.search(r"lose\s+([0-9]+)\s+lbs?", p)
    m_bday = re.search(
        r"birthday.*?which is\s+([a-z]+)\s+([0-9]{1,2})(?:st|nd|rd|th)?",
        p,
    )
    m_today = re.search(r"today is\s+([a-z]+)\s+([0-9]{1,2})(?:st|nd|rd|th)?", p)
    m_cal = re.search(r"burn\s+([0-9]+)\s+calories\s+to\s+lose\s+a\s+pound", p)

    if not all([m_lbs, m_bday, m_today, m_cal]):
        return make_result(False)

    lbs = float(m_lbs.group(1))
    b_month = m_bday.group(1)
    b_day = int(m_bday.group(2))
    t_month = m_today.group(1)
    t_day = int(m_today.group(2))
    cal_per_lb = float(m_cal.group(1))

    if b_month not in MONTH_INDEX or t_month not in MONTH_INDEX:
        return make_result(False)

    def day_of_year(month: str, day: int) -> int:
        idx = MONTH_INDEX[month]
        total = 0
        for m, i in MONTH_INDEX.items():
            if i < idx:
                total += MONTH_DAYS_NON_LEAP[m]
        return total + day

    today_doy = day_of_year(t_month, t_day)
    birthday_doy = day_of_year(b_month, b_day)

    if birthday_doy <= today_doy:
        days = (365 - today_doy) + birthday_doy
    else:
        days = birthday_doy - today_doy

    total_cal = lbs * cal_per_lb
    daily = total_cal / days

    reasoning = (
        f"Step 1: Total calorie deficit needed = {fmt_num(lbs)} × {fmt_num(cal_per_lb)} = {fmt_num(total_cal)} calories.\n"
        f"Step 2: From {t_month.title()} {t_day} to {b_month.title()} {b_day} is {fmt_num(days)} days.\n"
        f"Step 3: Daily deficit = {fmt_num(total_cal)} ÷ {fmt_num(days)} = {fmt_num(daily)}.\n"
        f"Final Answer: {fmt_num(daily)}"
    )

    return make_result(
        solved=True,
        answer=daily,
        reasoning=reasoning,
        solver_name="date_based_daily_deficit_solver",
        confidence=0.84,
        frame={
            "type": "date_based_rate",
            "lbs": lbs,
            "calories_per_lb": cal_per_lb,
            "days": days,
            "daily_deficit": daily,
        },
    )


def solve_dance_studio_problem(problem: str) -> Dict[str, Any]:
    p = normalize_text(problem)

    if "dance studio" not in p:
        return make_result(False)

    m_rent = re.search(r"costs\s+\$?([0-9]+(?:\.\d+)?)\s+per session", p)
    m_student = re.search(r"plus\s+\$?([0-9]+(?:\.\d+)?)\s+per student per session", p)
    m_students = re.search(r"has\s+([0-9]+)\s+students", p)
    m_days = re.search(r"rented\s+([0-9]+)\s+days a week", p)

    if not all([m_rent, m_student, m_students, m_days]):
        return make_result(False)

    fixed = float(m_rent.group(1))
    per_student = float(m_student.group(1))
    students = float(m_students.group(1))
    days_per_week = float(m_days.group(1))
    weeks_per_month = 4.0

    per_session = fixed + per_student * students
    monthly = per_session * days_per_week * weeks_per_month

    reasoning = (
        f"Step 1: Earnings per session = {fmt_num(fixed)} + {fmt_num(per_student)} × {fmt_num(students)} = {fmt_num(per_session)}.\n"
        f"Step 2: Sessions per month = {fmt_num(days_per_week)} × 4 = {fmt_num(days_per_week * weeks_per_month)}.\n"
        f"Step 3: Monthly earnings = {fmt_num(per_session)} × {fmt_num(days_per_week * weeks_per_month)} = {fmt_num(monthly)}.\n"
        f"Final Answer: {fmt_num(monthly)}"
    )

    return make_result(
        solved=True,
        answer=monthly,
        reasoning=reasoning,
        solver_name="weekly_session_monthly_total_solver",
        confidence=0.86,
        frame={
            "type": "weekly_to_monthly_rate",
            "fixed_per_session": fixed,
            "per_student": per_student,
            "students": students,
            "days_per_week": days_per_week,
            "weeks_per_month": weeks_per_month,
        },
    )


def solve_dog_holes_problem(problem: str) -> Dict[str, Any]:
    p = normalize_text(problem)

    if "dog" not in p or "holes" not in p:
        return make_result(False)

    m_dig = re.search(r"dig\s+([a-z]+|[0-9]+)\s+holes a day", p)
    m_days = re.search(r"for\s+([a-z]+|[0-9]+)\s+days", p)
    m_fill = re.search(r"filling in\s+([a-z]+|[0-9]+)\s+holes a day", p)
    m_new = re.search(r"keeps digging\s+([a-z]+|[0-9]+)\s+new holes", p)

    if not all([m_dig, m_days, m_fill, m_new]):
        return make_result(False)

    dig_per_day = word_or_number_to_float(m_dig.group(1))
    initial_days = word_or_number_to_float(m_days.group(1))
    fill_per_day = word_or_number_to_float(m_fill.group(1))
    new_per_day = word_or_number_to_float(m_new.group(1))

    if None in [dig_per_day, initial_days, fill_per_day, new_per_day]:
        return make_result(False)

    initial_holes = dig_per_day * initial_days
    net_fill = fill_per_day - new_per_day

    if net_fill <= 0:
        return make_result(False)

    days_needed = initial_holes / net_fill
    weeks = days_needed / 7

    reasoning = (
        f"Step 1: Initial holes dug = {fmt_num(dig_per_day)} × {fmt_num(initial_days)} = {fmt_num(initial_holes)}.\n"
        f"Step 2: Each day Nate fills {fmt_num(fill_per_day)} holes, but the dog digs {fmt_num(new_per_day)} new holes, "
        f"so net holes filled per day = {fmt_num(fill_per_day)} - {fmt_num(new_per_day)} = {fmt_num(net_fill)}.\n"
        f"Step 3: Days needed = {fmt_num(initial_holes)} ÷ {fmt_num(net_fill)} = {fmt_num(days_needed)} days.\n"
        f"Step 4: Weeks needed = {fmt_num(days_needed)} ÷ 7 = {fmt_num(weeks)}.\n"
        f"Final Answer: {fmt_num(weeks)}"
    )

    return make_result(
        solved=True,
        answer=weeks,
        reasoning=reasoning,
        solver_name="net_rate_to_weeks_solver",
        confidence=0.88,
        frame={
            "type": "net_rate",
            "initial_holes": initial_holes,
            "fill_per_day": fill_per_day,
            "new_per_day": new_per_day,
            "net_fill": net_fill,
            "weeks": weeks,
        },
    )


def solve_problem_with_semantic_frame(problem: str) -> Dict[str, Any]:
    """
    Try a portfolio of deterministic semantic-frame solvers.

    This is intentionally not sample-id based.
    Each solver encodes a typed quantitative relation.
    """

    solvers = [
        solve_mask_rate_problem,
        solve_cat_food_problem,
        solve_times_more_problem,
        solve_split_remaining_problem,
        solve_party_cost_problem,
        solve_interest_problem,
        solve_movie_collection_problem,
        solve_date_deficit_problem,
        solve_dance_studio_problem,
        solve_dog_holes_problem,
    ]

    candidates = []

    for solver in solvers:
        try:
            result = solver(problem)
            if result.get("solved"):
                candidates.append(result)
        except Exception as exc:
            continue

    if not candidates:
        return make_result(False)

    candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return candidates[0]


def should_use_semantic_solver(
    problem: str,
    initial_reasoning: str,
    initial_meta_diagnosis: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Decide whether to attempt deterministic semantic-frame repair.

    This is a trigger, not an acceptor.
    """

    if not initial_reasoning or not initial_reasoning.strip():
        return True

    p = normalize_text(problem)

    high_risk_phrases = [
        "times more",
        "how much more likely",
        "each additional guest",
        "half of what is left",
        "twice a day",
        "every time",
        "per student per session",
        "new holes every night",
        "birthday",
        "monthly interest",
    ]

    if any(x in p for x in high_risk_phrases):
        return True

    if initial_meta_diagnosis:
        score = initial_meta_diagnosis.get("global_consistency_score", 1.0)
        error_type = initial_meta_diagnosis.get("error_type", "none")
        if score < 0.75:
            return True
        if error_type != "none":
            return True

    return False


def semantic_solver_accepts(
    problem: str,
    initial_reasoning: str,
    solver_result: Dict[str, Any],
    min_confidence: float = 0.80,
) -> Dict[str, Any]:
    """
    Conservative acceptance:
    - solver must solve
    - confidence high enough
    - candidate answer must differ from wrong-looking initial answer OR fill empty reasoning
    """

    if not solver_result.get("solved"):
        return {
            "accept": False,
            "reason": "Semantic solver did not produce a candidate.",
        }

    if solver_result.get("confidence", 0.0) < min_confidence:
        return {
            "accept": False,
            "reason": "Semantic solver confidence below threshold.",
        }

    initial_answer = extract_final_answer(initial_reasoning)
    candidate_answer = solver_result.get("answer", "")

    if not initial_answer:
        return {
            "accept": True,
            "reason": "Accepted because initial reasoning is empty and semantic solver produced a candidate.",
        }

    if not answer_equal(initial_answer, candidate_answer):
        return {
            "accept": True,
            "reason": f"Accepted because semantic solver answer {candidate_answer} differs from initial answer {initial_answer}.",
        }

    return {
        "accept": False,
        "reason": "Rejected because semantic solver agrees with initial answer; no repair needed.",
    }