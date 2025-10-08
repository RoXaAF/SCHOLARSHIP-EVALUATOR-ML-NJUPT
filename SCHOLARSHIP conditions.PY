import pandas as pd
import numpy as np
import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import threading
from PIL import Image, ImageTk, ImageDraw
import urllib.request
import io

warnings.filterwarnings('ignore')

# ===================== ENUMS AND DATA STRUCTURES =====================

class MoralGrade(Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    QUALIFIED = "Qualified"
    UNQUALIFIED = "Unqualified"

class ScholarshipLevel(Enum):
    FIRST_CLASS = "First-Class"
    SECOND_CLASS = "Second-Class"
    THIRD_CLASS = "Third-Class"
    NOT_ELIGIBLE = "Not Eligible"

@dataclass
class StudentRecord:
    student_id: str
    name: str
    attendance_rate: float
    penalty_points: int
    important_absences: int
    cumulative_gpa: float
    previous_year_gpa: float
    failed_credits: int
    awards: List[str]
    publications: List[str]
    leadership_roles: List[str]
    fees_paid: bool
    has_serious_violations: bool
    interview_score: float
    semester: Optional[int] = None
    year: Optional[int] = None

# ===================== CORE EVALUATION ENGINE =====================

class NJUPTScholarshipEvaluator:
    def __init__(self, year: int):
        self.evaluation_year = year
        self.attendance_thresholds = self._get_attendance_thresholds()
    
    def _get_attendance_thresholds(self) -> Dict[ScholarshipLevel, float]:
        return {
            ScholarshipLevel.FIRST_CLASS: 95.0,
            ScholarshipLevel.SECOND_CLASS: 93.0,
            ScholarshipLevel.THIRD_CLASS: 90.0
        }
    
    def evaluate_moral_grade(self, student: StudentRecord) -> MoralGrade:
        if (student.attendance_rate >= 95.0 and
            student.penalty_points == 0 and 
            student.important_absences == 0):
            return MoralGrade.EXCELLENT
        elif (92.0 <= student.attendance_rate < 95.0 and 
              student.penalty_points <= 1 and 
              student.important_absences <= 1):
            return MoralGrade.GOOD
        elif (90.0 <= student.attendance_rate < 92.0 and 
              student.penalty_points <= 2 and 
              student.important_absences <= 2):
            return MoralGrade.QUALIFIED
        else:
            return MoralGrade.UNQUALIFIED
    
    def check_basic_eligibility(self, student: StudentRecord) -> Tuple[bool, List[str]]:
        issues = []
        if not student.fees_paid:
            issues.append("Outstanding fees not paid")
        if student.has_serious_violations:
            issues.append("Has serious disciplinary violations")
        if student.attendance_rate < 88.0:
            issues.append(f"Attendance rate {student.attendance_rate}% below minimum 88%")
        return len(issues) == 0, issues
    
    def check_scholarship_requirements(self, student: StudentRecord, level: ScholarshipLevel) -> Tuple[bool, List[str]]:
        issues = []
        min_attendance = self.attendance_thresholds[level]
        
        if student.attendance_rate < min_attendance:
            issues.append(f"Attendance {student.attendance_rate}% below required {min_attendance}%")
        
        if level in [ScholarshipLevel.FIRST_CLASS, ScholarshipLevel.SECOND_CLASS]:
            if student.failed_credits > 0:
                issues.append(f"Failed credits {student.failed_credits} exceeds maximum 0")
        elif level == ScholarshipLevel.THIRD_CLASS:
            if student.failed_credits > 3:
                issues.append(f"Failed credits {student.failed_credits} exceeds maximum 3")
        
        moral_grade = self.evaluate_moral_grade(student)
        if moral_grade == MoralGrade.UNQUALIFIED:
            issues.append("Moral and quality grade is unqualified")
        
        return len(issues) == 0, issues
    
    def evaluate_GPA_grade(self, student: StudentRecord) -> MoralGrade:
        if student.cumulative_gpa >= 3.5:
            return MoralGrade.EXCELLENT
        elif 3.2 <= student.cumulative_gpa < 3.5:
            return MoralGrade.GOOD
        elif 2.8 <= student.cumulative_gpa < 3.2:
            return MoralGrade.QUALIFIED
        else:
            return MoralGrade.UNQUALIFIED
    
    def calculate_academic_score(self, student: StudentRecord) -> float:
        return (student.cumulative_gpa * 0.8) + (student.previous_year_gpa * 0.2)
    
    def calculate_ability_score(self, student: StudentRecord) -> float:
        score = (len(student.awards) * 2 + 
                len(student.publications) * 5 + 
                len(student.leadership_roles) * 3)
        return min(score, 100)
    
    def calculate_total_score(self, student: StudentRecord) -> float:
        academic_score = self.calculate_academic_score(student)
        ability_score = self.calculate_ability_score(student)
        return (academic_score * 0.6 + ability_score * 0.2 + student.interview_score * 0.2)
    
    def evaluate_student(self, student: StudentRecord) -> Dict:
        moral_grade = self.evaluate_moral_grade(student)
        gpa_grade = self.evaluate_GPA_grade(student)
        basic_eligible, basic_issues = self.check_basic_eligibility(student)
        
        results = {
            'student_id': student.student_id,
            'name': student.name,
            'moral_grade': moral_grade.value,
            'gpa_grade': gpa_grade.value,
            'basic_eligible': basic_eligible,
            'basic_issues': basic_issues,
            'attendance_rate': student.attendance_rate,
            'academic_score': self.calculate_academic_score(student),
            'ability_score': self.calculate_ability_score(student),
            'total_score': self.calculate_total_score(student),
            'scholarship_levels': {}
        }
        
        if not basic_eligible:
            results['recommended_scholarship'] = ScholarshipLevel.NOT_ELIGIBLE.value
            return results
        
        for level in [ScholarshipLevel.FIRST_CLASS, ScholarshipLevel.SECOND_CLASS, ScholarshipLevel.THIRD_CLASS]:
            eligible, issues = self.check_scholarship_requirements(student, level)
            results['scholarship_levels'][level.value] = {
                'eligible': eligible,
                'issues': issues
            }
        
        recommended_level = ScholarshipLevel.NOT_ELIGIBLE
        for level in [ScholarshipLevel.FIRST_CLASS, ScholarshipLevel.SECOND_CLASS, ScholarshipLevel.THIRD_CLASS]:
            if results['scholarship_levels'][level.value]['eligible']:
                recommended_level = level
                break
        
        results['recommended_scholarship'] = recommended_level.value
        return results

# ===================== MACHINE LEARNING ENGINE =====================

class ScholarshipMLPredictor:
    def __init__(self):
        self.gpa_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scholarship_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attendance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.is_trained = False
        self.feature_importance = {}
    
    def prepare_features(self, students: List[StudentRecord]) -> np.ndarray:
        features = []
        for student in students:
            feature_vector = [
                student.attendance_rate,
                student.penalty_points,
                student.important_absences,
                student.cumulative_gpa,
                student.previous_year_gpa,
                student.failed_credits,
                len(student.awards),
                len(student.publications),
                len(student.leadership_roles),
                1 if student.fees_paid else 0,
                1 if student.has_serious_violations else 0,
                student.interview_score,
                student.cumulative_gpa - student.previous_year_gpa  # GPA trend
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def train_models(self, historical_data: pd.DataFrame) -> Dict:
        try:
            if len(historical_data) < 20:
                return {'success': False, 'error': 'Need at least 20 historical records for training'}
            
            # Prepare data
            students = self._dataframe_to_students(historical_data)
            X = self.prepare_features(students)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train GPA predictor
            y_gpa = historical_data['next_semester_gpa'].values
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_gpa, test_size=0.2, random_state=42)
            self.gpa_model.fit(X_train, y_train)
            gpa_score = self.gpa_model.score(X_test, y_test)
            
            # Train scholarship predictor
            y_scholarship = historical_data['scholarship_level'].values
            y_scholarship_encoded = self.label_encoder.fit_transform(y_scholarship)
            self.scholarship_model.fit(X_scaled, y_scholarship_encoded)
            scholarship_score = self.scholarship_model.score(X_scaled, y_scholarship_encoded)
            
            # Train risk predictor (at-risk = GPA drop > 0.3 or attendance < 90)
            y_risk = ((historical_data['next_semester_gpa'] - historical_data['cumulative_gpa'] < -0.3) | 
                     (historical_data['attendance_rate'] < 90)).astype(int).values
            self.risk_model.fit(X_scaled, y_risk)
            risk_score = self.risk_model.score(X_scaled, y_risk)
            
            # Train attendance predictor
            y_attendance = historical_data['next_semester_attendance'].values
            self.attendance_model.fit(X_scaled, y_attendance)
            attendance_score = self.attendance_model.score(X_scaled, y_attendance)
            
            self.is_trained = True
            
            # Calculate feature importance
            feature_names = ['Attendance', 'Penalty Points', 'Important Absences', 'Cumulative GPA', 
                           'Previous GPA', 'Failed Credits', 'Awards', 'Publications', 'Leadership',
                           'Fees Paid', 'Violations', 'Interview Score', 'GPA Trend']
            self.feature_importance = dict(zip(feature_names, self.gpa_model.feature_importances_))
            
            return {
                'success': True,
                'gpa_r2': round(gpa_score, 3),
                'scholarship_accuracy': round(scholarship_score, 3),
                'risk_accuracy': round(risk_score, 3),
                'attendance_r2': round(attendance_score, 3),
                'training_samples': len(historical_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_future_performance(self, student: StudentRecord) -> Dict:
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        try:
            features = self.prepare_features([student])
            features_scaled = self.scaler.transform(features)
            
            # Predictions
            predicted_gpa = float(self.gpa_model.predict(features_scaled)[0])
            predicted_scholarship = self.label_encoder.inverse_transform(
                self.scholarship_model.predict(features_scaled)
            )[0]
            risk_probability = float(self.risk_model.predict_proba(features_scaled)[0][1])
            predicted_attendance = float(self.attendance_model.predict(features_scaled)[0])
            
            # Confidence scores
            scholarship_proba = self.scholarship_model.predict_proba(features_scaled)[0]
            scholarship_confidence = float(max(scholarship_proba))
            
            return {
                'predicted_gpa': round(predicted_gpa, 2),
                'predicted_scholarship': predicted_scholarship,
                'scholarship_confidence': round(scholarship_confidence * 100, 1),
                'at_risk_probability': round(risk_probability * 100, 1),
                'predicted_attendance': round(predicted_attendance, 1),
                'gpa_trend': round(predicted_gpa - student.cumulative_gpa, 2),
                'recommendations': self._generate_recommendations(student, predicted_gpa, risk_probability)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, student: StudentRecord, predicted_gpa: float, risk: float) -> List[str]:
        recommendations = []
        
        if risk > 0.5:
            recommendations.append("⚠️ High risk detected - consider academic counseling")
        
        if student.attendance_rate < 92:
            recommendations.append("📚 Improve attendance to increase scholarship eligibility")
        
        if predicted_gpa < student.cumulative_gpa:
            recommendations.append("📉 GPA may decline - focus on core subjects")
        
        if len(student.publications) == 0:
            recommendations.append("📝 Consider research publications to boost ability score")
        
        if student.penalty_points > 0:
            recommendations.append("✅ Maintain zero penalty points for Excellent moral grade")
        
        if not recommendations:
            recommendations.append("🎯 Keep up the excellent work!")
        
        return recommendations
    
    def _dataframe_to_students(self, df: pd.DataFrame) -> List[StudentRecord]:
        students = []
        for _, row in df.iterrows():
            awards = str(row.get('awards', '')).split(';') if pd.notna(row.get('awards')) else []
            publications = str(row.get('publications', '')).split(';') if pd.notna(row.get('publications')) else []
            leadership = str(row.get('leadership_roles', '')).split(';') if pd.notna(row.get('leadership_roles')) else []
            
            student = StudentRecord(
                student_id=str(row['student_id']),
                name=str(row['name']),
                attendance_rate=float(row['attendance_rate']),
                penalty_points=int(row['penalty_points']),
                important_absences=int(row['important_absences']),
                cumulative_gpa=float(row['cumulative_gpa']),
                previous_year_gpa=float(row['previous_year_gpa']),
                failed_credits=int(row['failed_credits']),
                awards=[a.strip() for a in awards if a.strip()],
                publications=[p.strip() for p in publications if p.strip()],
                leadership_roles=[l.strip() for l in leadership if l.strip()],
                fees_paid=str(row.get('fees_paid', 'YES')).upper() in ['YES', 'TRUE', '1', 'Y'],
                has_serious_violations=str(row.get('has_serious_violations', 'NO')).upper() in ['YES', 'TRUE', '1', 'Y'],
                interview_score=float(row['interview_score'])
            )
            students.append(student)
        return students
    
    def save_models(self, filepath: str = 'ml_models.pkl'):
        if not self.is_trained:
            return False
        try:
            models_data = {
                'gpa_model': self.gpa_model,
                'scholarship_model': self.scholarship_model,
                'risk_model': self.risk_model,
                'attendance_model': self.attendance_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_importance': self.feature_importance
            }
            with open(filepath, 'wb') as f:
                pickle.dump(models_data, f)
            return True
        except:
            return False
    
    def load_models(self, filepath: str = 'ml_models.pkl'):
        try:
            with open(filepath, 'rb') as f:
                models_data = pickle.load(f)
            self.gpa_model = models_data['gpa_model']
            self.scholarship_model = models_data['scholarship_model']
            self.risk_model = models_data['risk_model']
            self.attendance_model = models_data['attendance_model']
            self.scaler = models_data['scaler']
            self.label_encoder = models_data['label_encoder']
            self.feature_importance = models_data['feature_importance']
            self.is_trained = True
            return True
        except:
            return False

# ===================== EXCEL MANAGER =====================

class ExcelManager:
    @staticmethod
    def create_template():
        template_data = {
            'student_id': ['B12345678', ''],
            'name': ['Zhang Wei', ''],
            'attendance_rate': [96.5, ''],
            'penalty_points': [0, ''],
            'important_absences': [0, ''],
            'cumulative_gpa': [3.8, ''],
            'previous_year_gpa': [3.7, ''],
            'failed_credits': [0, ''],
            'awards': ['National Award; Provincial Award', ''],
            'publications': ['IEEE Paper; ACM Paper', ''],
            'leadership_roles': ['Class Monitor; Club President', ''],
            'fees_paid': ['YES', ''],
            'has_serious_violations': ['NO', ''],
            'interview_score': [85.0, '']
        }
        
        df = pd.DataFrame(template_data)
        filename = 'student_data.xlsx'
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Student Data', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Student Data']
            
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#2B96EE', 'font_color': 'white', 
                'border': 1, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter'
            })
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            worksheet.set_column('A:A', 12)
            worksheet.set_column('B:B', 15)
            worksheet.set_column('C:C', 15)
            worksheet.set_column('D:D', 14)
            worksheet.set_column('E:E', 18)
            worksheet.set_column('F:G', 15)
            worksheet.set_column('H:H', 14)
            worksheet.set_column('I:K', 30)
            worksheet.set_column('L:M', 20)
            worksheet.set_column('N:N', 15)
        
        return filename
    
    @staticmethod
    def create_ml_training_template():
        template_data = {
            'student_id': ['B12345678', 'B87654321'],
            'name': ['Zhang Wei', 'Li Ming'],
            'attendance_rate': [96.5, 91.0],
            'penalty_points': [0, 1],
            'important_absences': [0, 1],
            'cumulative_gpa': [3.8, 3.3],
            'previous_year_gpa': [3.7, 3.2],
            'failed_credits': [0, 0],
            'awards': ['National Award; Provincial Award', 'Provincial Award'],
            'publications': ['IEEE Paper; ACM Paper', ''],
            'leadership_roles': ['Class Monitor; Club President', 'Class Monitor'],
            'fees_paid': ['YES', 'YES'],
            'has_serious_violations': ['NO', 'NO'],
            'interview_score': [85.0, 78.0],
            'next_semester_gpa': [3.85, 3.25],
            'next_semester_attendance': [97.0, 90.5],
            'scholarship_level': ['First-Class', 'Second-Class']
        }
        
        df = pd.DataFrame(template_data)
        filename = 'ml_training_data.xlsx'
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Training Data', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Training Data']
            
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#FF6B6B', 'font_color': 'white',
                'border': 1, 'text_wrap': True, 'align': 'center'
            })
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            for col in range(17):
                worksheet.set_column(col, col, 18)
        
        return filename
    
    @staticmethod
    def load_student_data(filename: str) -> List[StudentRecord]:
        if not os.path.exists(filename):
            return []
        
        try:
            df = pd.read_excel(filename)
            students = []
            
            for _, row in df.iterrows():
                if pd.isna(row['student_id']) or str(row['student_id']).strip() == '':
                    continue
                
                try:
                    awards = [a.strip() for a in str(row['awards']).split(';')] if pd.notna(row['awards']) else []
                    publications = [p.strip() for p in str(row['publications']).split(';')] if pd.notna(row['publications']) else []
                    leadership_roles = [r.strip() for r in str(row['leadership_roles']).split(';')] if pd.notna(row['leadership_roles']) else []
                    
                    student = StudentRecord(
                        student_id=str(row['student_id']).strip(),
                        name=str(row['name']).strip(),
                        attendance_rate=float(row['attendance_rate']),
                        penalty_points=int(row['penalty_points']),
                        important_absences=int(row['important_absences']),
                        cumulative_gpa=float(row['cumulative_gpa']),
                        previous_year_gpa=float(row['previous_year_gpa']),
                        failed_credits=int(row['failed_credits']),
                        awards=[a for a in awards if a],
                        publications=[p for p in publications if p],
                        leadership_roles=[r for r in leadership_roles if r],
                        fees_paid=str(row['fees_paid']).upper() in ['YES', 'TRUE', '1', 'Y'],
                        has_serious_violations=str(row['has_serious_violations']).upper() in ['YES', 'TRUE', '1', 'Y'],
                        interview_score=float(row['interview_score'])
                    )
                    students.append(student)
                except:
                    continue
            
            return students
        except:
            return []
    
    @staticmethod
    def save_results(results: List[Dict], filename: str = 'scholarship_results.xlsx'):
        output_data = []
        for result in results:
            output_row = {
                'Student ID': result['student_id'],
                'Name': result['name'],
                'Attendance Rate': f"{result['attendance_rate']}%",
                'Moral Grade': result['moral_grade'],
                'GPA Grade': result['gpa_grade'],
                'Basic Eligible': 'YES' if result['basic_eligible'] else 'NO',
                'Recommended Scholarship': result['recommended_scholarship'],
                'Academic Score': round(result['academic_score'], 2),
                'Ability Score': round(result['ability_score'], 2),
                'Total Score': round(result['total_score'], 2),
                'First Class Eligible': 'YES' if result.get('scholarship_levels', {}).get('First-Class', {}).get('eligible', False) else 'NO',
                'Second Class Eligible': 'YES' if result.get('scholarship_levels', {}).get('Second-Class', {}).get('eligible', False) else 'NO',
                'Third Class Eligible': 'YES' if result.get('scholarship_levels', {}).get('Third-Class', {}).get('eligible', False) else 'NO',
                'Issues': ' | '.join(result['basic_issues']) if result['basic_issues'] else 'None'
            }
            output_data.append(output_row)
        
        df = pd.DataFrame(output_data)
        df.to_excel(filename, index=False)

# ===================== GUI APPLICATION =====================

class ScholarshipGUI:
    # NJUPT Brand Colors
    NJUPT_BLUE = '#1E4D92'  # Main NJUPT Blue
    NJUPT_WHITE = '#FFFFFF'  # White
    ACCENT_LIGHT = '#3A6BB5'  # Lighter blue
    ACCENT_DARK = '#0F2E5C'  # Darker blue
    TEXT_DARK = '#2C3E50'
    BG_LIGHT = '#F8F9FA'
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 NJUPT Scholarship Evaluation & ML Prediction System")
        
        # Start in fullscreen
        self.root.state('zoomed')  # Windows
        try:
            self.root.attributes('-zoomed', True)  # Linux
        except:
            pass
        
        # Allow toggling fullscreen with F11
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)
        
        self.root.configure(bg=self.NJUPT_BLUE)
        
        self.evaluator = None
        self.ml_predictor = ScholarshipMLPredictor()
        self.current_results = []
        self.current_students = []
        
        # Configure custom style
        self.setup_styles()
        
        # Load logo
        self.logo_image = self.load_logo()
        
        self.setup_ui()
        self.load_saved_models()
    
    def toggle_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', True)
        return 'break'
    
    def exit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)
        return 'break'
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure main colors
        style.configure('TFrame', background=self.NJUPT_WHITE)
        style.configure('TLabel', background=self.NJUPT_WHITE, foreground=self.TEXT_DARK, 
                       font=('Segoe UI', 10))
        style.configure('TButton', background=self.NJUPT_BLUE, foreground=self.NJUPT_WHITE,
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, relief='flat')
        style.map('TButton',
                 background=[('active', self.ACCENT_LIGHT), ('pressed', self.ACCENT_DARK)])
        
        # Notebook (tabs) styling
        style.configure('TNotebook', background=self.NJUPT_BLUE, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.ACCENT_LIGHT, foreground=self.NJUPT_WHITE,
                       padding=[20, 10], font=('Segoe UI', 11, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', self.NJUPT_BLUE)],
                 foreground=[('selected', self.NJUPT_WHITE)])
        
        # LabelFrame styling
        style.configure('TLabelframe', background=self.NJUPT_WHITE, borderwidth=2,
                       relief='solid', bordercolor=self.NJUPT_BLUE)
        style.configure('TLabelframe.Label', background=self.NJUPT_WHITE, 
                       foreground=self.NJUPT_BLUE, font=('Segoe UI', 12, 'bold'))
        
        # Entry styling
        style.configure('TEntry', fieldbackground=self.NJUPT_WHITE, 
                       foreground=self.TEXT_DARK, borderwidth=2)
        
        # Combobox styling
        style.configure('TCombobox', fieldbackground=self.NJUPT_WHITE,
                       background=self.NJUPT_BLUE, foreground=self.TEXT_DARK)
        
        # Treeview styling
        style.configure('Treeview', background=self.NJUPT_WHITE, 
                       foreground=self.TEXT_DARK, fieldbackground=self.NJUPT_WHITE,
                       font=('Segoe UI', 9))
        style.configure('Treeview.Heading', background=self.NJUPT_BLUE,
                       foreground=self.NJUPT_WHITE, font=('Segoe UI', 10, 'bold'))
        style.map('Treeview.Heading',
                 background=[('active', self.ACCENT_LIGHT)])
    
    def load_logo(self):
        """Load and process the NJUPT logo with transparency"""
        try:
            # Download logo from the uploaded image URL
            logo_url = "https://upload.wikimedia.org/wikipedia/en/thumb/4/4a/Nanjing_University_of_Posts_and_Telecommunications_logo.svg/1200px-Nanjing_University_of_Posts_and_Telecommunications_logo.svg.png"
            
            with urllib.request.urlopen(logo_url) as url:
                image_data = url.read()
            
            # Open and process image
            img = Image.open(io.BytesIO(image_data))
            
            # Resize to appropriate size (200x200)
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            
            # Apply 10% transparency (90% opacity)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Adjust alpha channel to 90% (230 out of 255)
            alpha = img.split()[3]
            alpha = alpha.point(lambda p: int(p * 0.9))
            img.putalpha(alpha)
            
            return ImageTk.PhotoImage(img)
            
        except Exception as e:
            print(f"Could not load logo: {e}")
            # Create a placeholder logo
            img = Image.new('RGBA', (200, 200), (30, 77, 146, 230))
            draw = ImageDraw.Draw(img)
            draw.text((70, 90), "NJUPT", fill=(255, 255, 255, 230))
            return ImageTk.PhotoImage(img)
    
    def setup_ui(self):
        # Main container with NJUPT blue background
        main_container = tk.Frame(self.root, bg=self.NJUPT_BLUE)
        main_container.pack(fill='both', expand=True)
        
        # Header with logo
        header_frame = tk.Frame(main_container, bg=self.NJUPT_BLUE, height=120)
        header_frame.pack(fill='x', padx=20, pady=20)
        header_frame.pack_propagate(False)
        
        # Logo in the center
        if self.logo_image:
            logo_label = tk.Label(header_frame, image=self.logo_image, bg=self.NJUPT_BLUE)
            logo_label.pack(side='top', pady=10)
        
        # Title
        title_label = tk.Label(header_frame, 
                               text="南京邮电大学 奖学金评估系统",
                               font=('Microsoft YaHei UI', 24, 'bold'),
                               bg=self.NJUPT_BLUE, fg=self.NJUPT_WHITE)
        title_label.pack(side='top', pady=5)
        
        subtitle_label = tk.Label(header_frame,
                                 text="NJUPT Scholarship Evaluation & ML Prediction System",
                                 font=('Segoe UI', 14),
                                 bg=self.NJUPT_BLUE, fg=self.NJUPT_WHITE)
        subtitle_label.pack(side='top')
        
        # Content area with white background
        content_frame = tk.Frame(main_container, bg=self.NJUPT_WHITE)
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Notebook (tabs) inside content frame
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab_evaluation = ttk.Frame(self.notebook)
        self.tab_ml_training = ttk.Frame(self.notebook)
        self.tab_ml_prediction = ttk.Frame(self.notebook)
        self.tab_analytics = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_evaluation, text='📊 评估 Evaluation')
        self.notebook.add(self.tab_ml_training, text='🧠 机器学习训练 ML Training')
        self.notebook.add(self.tab_ml_prediction, text='🔮 预测 Predictions')
        self.notebook.add(self.tab_analytics, text='📈 分析 Analytics')
        
        self.setup_evaluation_tab()
        self.setup_ml_training_tab()
        self.setup_ml_prediction_tab()
        self.setup_analytics_tab()
    
    def create_button(self, parent, text, command):
        """Create a styled button with NJUPT colors"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=self.NJUPT_BLUE, fg=self.NJUPT_WHITE,
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat', borderwidth=0,
                       padx=20, pady=10, cursor='hand2',
                       activebackground=self.ACCENT_LIGHT,
                       activeforeground=self.NJUPT_WHITE)
        
        # Hover effects
        btn.bind('<Enter>', lambda e: e.widget.config(bg=self.ACCENT_LIGHT))
        btn.bind('<Leave>', lambda e: e.widget.config(bg=self.NJUPT_BLUE))
        
        return btn
    
    def setup_ui(self):
        # Main container with notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab_evaluation = ttk.Frame(self.notebook)
        self.tab_ml_training = ttk.Frame(self.notebook)
        self.tab_ml_prediction = ttk.Frame(self.notebook)
        self.tab_analytics = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_evaluation, text='📊 Evaluation')
        self.notebook.add(self.tab_ml_training, text='🧠 ML Training')
        self.notebook.add(self.tab_ml_prediction, text='🔮 Predictions')
        self.notebook.add(self.tab_analytics, text='📈 Analytics')
        
        self.setup_evaluation_tab()
        self.setup_ml_training_tab()
        self.setup_ml_prediction_tab()
        self.setup_analytics_tab()
    
    def setup_evaluation_tab(self):
        # Top frame - Controls
        control_frame = ttk.LabelFrame(self.tab_evaluation, text="控制面板 Controls", padding=15)
        control_frame.pack(fill='x', padx=15, pady=15)
        
        # Grid layout for controls
        ttk.Label(control_frame, text="评估年份 Evaluation Year:", 
                 font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.year_entry = ttk.Entry(control_frame, width=15, font=('Segoe UI', 11))
        self.year_entry.insert(0, str(datetime.now().year))
        self.year_entry.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        # Buttons with custom styling
        btn_frame = tk.Frame(control_frame, bg=self.NJUPT_WHITE)
        btn_frame.grid(row=0, column=2, columnspan=4, padx=10, pady=5)
        
        self.create_button(btn_frame, "📄 创建模板 Create Template", 
                          self.create_template).pack(side='left', padx=5)
        self.create_button(btn_frame, "📂 加载数据 Load Data", 
                          self.load_data).pack(side='left', padx=5)
        self.create_button(btn_frame, "▶️ 评估所有 Evaluate All", 
                          self.evaluate_all).pack(side='left', padx=5)
        self.create_button(btn_frame, "💾 导出结果 Export Results", 
                          self.export_results).pack(side='left', padx=5)
        
        # Results table
        table_frame = ttk.LabelFrame(self.tab_evaluation, text="评估结果 Results", padding=15)
        table_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Create treeview with scrollbars
        tree_container = tk.Frame(table_frame, bg=self.NJUPT_WHITE)
        tree_container.pack(fill='both', expand=True)
        
        v_scroll = ttk.Scrollbar(tree_container, orient='vertical')
        h_scroll = ttk.Scrollbar(tree_container, orient='horizontal')
        
        columns = ('ID', 'Name', 'Attendance', 'Moral', 'GPA Grade', 'Scholarship', 'Total Score')
        self.results_tree = ttk.Treeview(tree_container, columns=columns, show='headings',
                                         yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set,
                                         height=20)
        
        v_scroll.config(command=self.results_tree.yview)
        h_scroll.config(command=self.results_tree.xview)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150, anchor='center')
        
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        
        # Alternating row colors
        self.results_tree.tag_configure('oddrow', background='#F8F9FA')
        self.results_tree.tag_configure('evenrow', background=self.NJUPT_WHITE)
        
        # Status bar
        self.status_label = tk.Label(self.tab_evaluation, text="✅ 系统就绪 Ready", 
                                     relief='flat', bg=self.NJUPT_BLUE, fg=self.NJUPT_WHITE,
                                     font=('Segoe UI', 10, 'bold'), pady=8)
        self.status_label.pack(fill='x', padx=15, pady=(0, 15))
    
    def setup_ml_training_tab(self):
        # Instructions
        inst_frame = ttk.LabelFrame(self.tab_ml_training, text="使用说明 Instructions", padding=15)
        inst_frame.pack(fill='x', padx=15, pady=15)
        
        instructions = ("📚 训练机器学习模型预测未来学生表现 Train ML models to predict future student performance\n\n"
                       "1. 创建机器学习训练模板（包含历史数据）Create ML training template (includes historical data)\n"
                       "2. 填写学生数据和下学期结果 Fill in student data + next semester outcomes\n"
                       "3. 加载数据并训练模型 Load the data and train models\n"
                       "4. 模型将预测：GPA、奖学金级别、风险状态 Models will predict: GPA, Scholarship Level, At-Risk Status")
        
        inst_label = tk.Label(inst_frame, text=instructions, justify='left',
                             bg=self.NJUPT_WHITE, fg=self.TEXT_DARK,
                             font=('Segoe UI', 10))
        inst_label.pack(fill='x')
        
        # Control buttons
        btn_frame = tk.Frame(self.tab_ml_training, bg=self.NJUPT_WHITE)
        btn_frame.pack(fill='x', padx=15, pady=10)
        
        self.create_button(btn_frame, "📄 创建ML模板 Create ML Template", 
                          self.create_ml_template).pack(side='left', padx=5)
        self.create_button(btn_frame, "📂 加载训练数据 Load Training Data", 
                          self.load_training_data).pack(side='left', padx=5)
        self.create_button(btn_frame, "🚀 训练模型 Train Models", 
                          self.train_models).pack(side='left', padx=5)
        self.create_button(btn_frame, "💾 保存模型 Save Models", 
                          self.save_models).pack(side='left', padx=5)
        self.create_button(btn_frame, "📥 加载模型 Load Models", 
                          self.load_models).pack(side='left', padx=5)
        
        # Training results
        result_frame = ttk.LabelFrame(self.tab_ml_training, text="训练结果 Training Results", padding=15)
        result_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.training_text = scrolledtext.ScrolledText(result_frame, height=20, wrap='word',
                                                       bg=self.NJUPT_WHITE, fg=self.TEXT_DARK,
                                                       font=('Consolas', 10),
                                                       relief='flat', borderwidth=2)
        self.training_text.pack(fill='both', expand=True)
        
        # Model status
        self.model_status_label = tk.Label(self.tab_ml_training, 
                                           text="❌ 模型未训练 Models not trained", 
                                           relief='flat', bg='#E74C3C', fg=self.NJUPT_WHITE,
                                           font=('Segoe UI', 10, 'bold'), pady=8)
        self.model_status_label.pack(fill='x', padx=15, pady=(0, 15))
    
    def setup_ml_prediction_tab(self):
        # Student selection
        select_frame = ttk.LabelFrame(self.tab_ml_prediction, text="选择学生 Select Student", padding=15)
        select_frame.pack(fill='x', padx=15, pady=15)
        
        ttk.Label(select_frame, text="学生 Student:", 
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=10)
        self.student_combo = ttk.Combobox(select_frame, width=50, state='readonly',
                                         font=('Segoe UI', 10))
        self.student_combo.pack(side='left', padx=10)
        
        self.create_button(select_frame, "🔮 预测表现 Predict Performance", 
                          self.predict_performance).pack(side='left', padx=10)
        
        # Prediction results
        pred_frame = ttk.LabelFrame(self.tab_ml_prediction, text="预测结果 Predictions", padding=15)
        pred_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.prediction_text = scrolledtext.ScrolledText(pred_frame, height=25, wrap='word',
                                                        bg=self.NJUPT_WHITE, fg=self.TEXT_DARK,
                                                        font=('Consolas', 10),
                                                        relief='flat', borderwidth=2)
        self.prediction_text.pack(fill='both', expand=True)
    
    def setup_analytics_tab(self):
        # Control frame
        control_frame = tk.Frame(self.tab_analytics, bg=self.NJUPT_WHITE)
        control_frame.pack(fill='x', padx=15, pady=15)
        
        ttk.Label(control_frame, text="选择图表类型 Select Chart Type:", 
                 font=('Segoe UI', 11, 'bold')).pack(side='left', padx=10)
        
        self.create_button(control_frame, "📊 GPA分布 GPA Distribution", 
                          lambda: self.show_chart('gpa')).pack(side='left', padx=5)
        self.create_button(control_frame, "📈 奖学金分布 Scholarship Distribution", 
                          lambda: self.show_chart('scholarship')).pack(side='left', padx=5)
        self.create_button(control_frame, "📉 出勤分析 Attendance Analysis", 
                          lambda: self.show_chart('attendance')).pack(side='left', padx=5)
        self.create_button(control_frame, "🎯 分数分解 Score Breakdown", 
                          lambda: self.show_chart('scores')).pack(side='left', padx=5)
        self.create_button(control_frame, "🔥 特征重要性 Feature Importance", 
                          lambda: self.show_chart('importance')).pack(side='left', padx=5)
        
        # Chart canvas
        self.chart_frame = ttk.Frame(self.tab_analytics)
        self.chart_frame.pack(fill='both', expand=True, padx=15, pady=15) 
    
    def setup_analytics_tab(self):
        # Control frame
        control_frame = ttk.Frame(self.tab_analytics)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="📊 GPA Distribution", command=lambda: self.show_chart('gpa')).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📈 Scholarship Distribution", command=lambda: self.show_chart('scholarship')).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📉 Attendance Analysis", command=lambda: self.show_chart('attendance')).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🎯 Score Breakdown", command=lambda: self.show_chart('scores')).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🔥 Feature Importance", command=lambda: self.show_chart('importance')).pack(side='left', padx=5)
        
        # Chart canvas
        self.chart_frame = ttk.Frame(self.tab_analytics)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # ===================== EVALUATION TAB METHODS =====================
    
    def create_template(self):
        try:
            filename = ExcelManager.create_template()
            messagebox.showinfo("Success", f"Template '{filename}' created!\n\nPlease:\n1. Open the file\n2. Enter student data\n3. Save and come back to load it")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create template: {e}")
    
    def load_data(self):
        filename = filedialog.askopenfilename(
            title="Select Student Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.current_students = ExcelManager.load_student_data(filename)
            
            if not self.current_students:
                messagebox.showwarning("Warning", "No valid student data found in file")
                return
            
            # Update student combo box for predictions
            student_names = [f"{s.student_id} - {s.name}" for s in self.current_students]
            self.student_combo['values'] = student_names
            
            self.status_label.config(text=f"Loaded {len(self.current_students)} students")
            messagebox.showinfo("Success", f"Loaded {len(self.current_students)} student(s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def evaluate_all(self):
        if not self.current_students:
            messagebox.showwarning("Warning", "Please load student data first")
            return
        
        try:
            year = int(self.year_entry.get())
            self.evaluator = NJUPTScholarshipEvaluator(year)
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            self.current_results = []
            
            # Evaluate each student
            for student in self.current_students:
                result = self.evaluator.evaluate_student(student)
                self.current_results.append(result)
                
                # Add to tree
                self.results_tree.insert('', 'end', values=(
                    result['student_id'],
                    result['name'],
                    f"{result['attendance_rate']}%",
                    result['moral_grade'],
                    result['gpa_grade'],
                    result['recommended_scholarship'],
                    f"{result['total_score']:.2f}"
                ))
            
            self.status_label.config(text=f"Evaluated {len(self.current_results)} students")
            messagebox.showinfo("Success", f"Evaluation complete for {len(self.current_results)} students")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year")
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {e}")
    
    def export_results(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export. Please evaluate students first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile="scholarship_results.xlsx"
        )
        
        if not filename:
            return
        
        try:
            ExcelManager.save_results(self.current_results, filename)
            messagebox.showinfo("Success", f"Results exported to '{filename}'")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    # ===================== ML TRAINING TAB METHODS =====================
    
    def create_ml_template(self):
        try:
            filename = ExcelManager.create_ml_training_template()
            messagebox.showinfo("Success", 
                              f"ML training template '{filename}' created!\n\n"
                              "Please fill in:\n"
                              "- Current student data\n"
                              "- next_semester_gpa\n"
                              "- next_semester_attendance\n"
                              "- scholarship_level\n\n"
                              "Need at least 20 records for training")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create template: {e}")
    
    def load_training_data(self):
        filename = filedialog.askopenfilename(
            title="Select ML Training Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.training_data = pd.read_excel(filename)
            
            required_cols = ['next_semester_gpa', 'next_semester_attendance', 'scholarship_level']
            missing = [col for col in required_cols if col not in self.training_data.columns]
            
            if missing:
                messagebox.showerror("Error", f"Missing required columns: {', '.join(missing)}")
                return
            
            self.training_text.insert('end', f"✅ Loaded {len(self.training_data)} training records\n")
            self.training_text.insert('end', f"Columns: {', '.join(self.training_data.columns)}\n\n")
            
            messagebox.showinfo("Success", f"Loaded {len(self.training_data)} training records")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load training data: {e}")
    
    def train_models(self):
        if not hasattr(self, 'training_data'):
            messagebox.showwarning("Warning", "Please load training data first")
            return
        
        self.training_text.insert('end', "\n" + "="*60 + "\n")
        self.training_text.insert('end', "🚀 Starting model training...\n")
        self.training_text.insert('end', "="*60 + "\n\n")
        
        def train_thread():
            try:
                result = self.ml_predictor.train_models(self.training_data)
                
                if result['success']:
                    self.training_text.insert('end', "✅ 训练成功完成！ Training completed successfully!\n\n")
                    self.training_text.insert('end', f"📊 训练样本数 Training Samples: {result['training_samples']}\n\n")
                    self.training_text.insert('end', "模型性能 Model Performance:\n")
                    self.training_text.insert('end', f"  • GPA预测器 R²: {result['gpa_r2']:.3f}\n")
                    self.training_text.insert('end', f"  • 奖学金分类器准确率 Accuracy: {result['scholarship_accuracy']:.3f}\n")
                    self.training_text.insert('end', f"  • 风险预测器准确率 Accuracy: {result['risk_accuracy']:.3f}\n")
                    self.training_text.insert('end', f"  • 出勤率预测器 R²: {result['attendance_r2']:.3f}\n\n")
                    
                    # Feature importance
                    if self.ml_predictor.feature_importance:
                        self.training_text.insert('end', "🎯 特征重要性 Top Feature Importances:\n")
                        sorted_features = sorted(self.ml_predictor.feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
                        for feat, imp in sorted_features:
                            self.training_text.insert('end', f"  • {feat}: {imp:.3f}\n")
                    
                    self.model_status_label.config(text="✅ 模型已训练就绪 Models trained and ready", 
                                                  bg='#27AE60')
                    messagebox.showinfo("成功 Success", "模型训练成功！ Models trained successfully!")
                else:
                    self.training_text.insert('end', f"❌ 训练失败 Training failed: {result['error']}\n")
                    messagebox.showerror("错误 Error", f"训练失败 Training failed: {result['error']}")
                
            except Exception as e:
                self.training_text.insert('end', f"❌ Error: {str(e)}\n")
                messagebox.showerror("Error", f"Training error: {e}")
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def save_models(self):
        if not self.ml_predictor.is_trained:
            messagebox.showwarning("Warning", "No trained models to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialfile="ml_models.pkl"
        )
        
        if not filename:
            return
        
        if self.ml_predictor.save_models(filename):
            messagebox.showinfo("Success", f"Models saved to '{filename}'")
        else:
            messagebox.showerror("Error", "Failed to save models")
    
    def load_models(self):
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        if self.ml_predictor.load_models(filename):
            self.model_status_label.config(text="✅ Models loaded and ready", foreground='green')
            messagebox.showinfo("Success", "Models loaded successfully!")
        else:
            messagebox.showerror("Error", "Failed to load models")
    
    def load_saved_models(self):
        """Try to load models on startup"""
        if os.path.exists('ml_models.pkl'):
            if self.ml_predictor.load_models('ml_models.pkl'):
                self.model_status_label.config(text="✅ 模型已加载就绪 Models loaded and ready", 
                                              bg='#27AE60')
    
    # ===================== ML PREDICTION TAB METHODS =====================
    
    def predict_performance(self):
        if not self.ml_predictor.is_trained:
            messagebox.showwarning("警告 Warning", "请先训练或加载ML模型\nPlease train or load ML models first")
            return
        
        if not self.current_students:
            messagebox.showwarning("警告 Warning", "请先加载学生数据\nPlease load student data first")
            return
        
        selected_idx = self.student_combo.current()
        if selected_idx < 0:
            messagebox.showwarning("警告 Warning", "请选择一个学生\nPlease select a student")
            return
        
        student = self.current_students[selected_idx]
        
        try:
            prediction = self.ml_predictor.predict_future_performance(student)
            
            if 'error' in prediction:
                messagebox.showerror("错误 Error", f"预测失败 Prediction failed: {prediction['error']}")
                return
            
            # Clear and display results
            self.prediction_text.delete('1.0', 'end')
            
            self.prediction_text.insert('end', "="*70 + "\n")
            self.prediction_text.insert('end', f"🔮 学生表现预测 PERFORMANCE PREDICTION FOR {student.name}\n")
            self.prediction_text.insert('end', "="*70 + "\n\n")
            
            self.prediction_text.insert('end', "📊 当前状态 CURRENT STATUS:\n")
            self.prediction_text.insert('end', f"  • 当前GPA Current GPA: {student.cumulative_gpa:.2f}\n")
            self.prediction_text.insert('end', f"  • 当前出勤率 Current Attendance: {student.attendance_rate}%\n")
            self.prediction_text.insert('end', f"  • 去年GPA Previous Year GPA: {student.previous_year_gpa:.2f}\n")
            self.prediction_text.insert('end', f"  • 处分分数 Penalty Points: {student.penalty_points}\n\n")
            
            self.prediction_text.insert('end', "🔮 预测结果 PREDICTIONS:\n")
            self.prediction_text.insert('end', f"  • 预测下学期GPA Predicted Next Semester GPA: {prediction['predicted_gpa']:.2f}\n")
            
            trend_icon = "📈" if prediction['gpa_trend'] > 0 else "📉" if prediction['gpa_trend'] < 0 else "➡️"
            self.prediction_text.insert('end', f"  • GPA趋势 GPA Trend: {trend_icon} {prediction['gpa_trend']:+.2f}\n")
            
            self.prediction_text.insert('end', f"  • 预测出勤率 Predicted Attendance: {prediction['predicted_attendance']:.1f}%\n")
            self.prediction_text.insert('end', f"  • 预测奖学金级别 Predicted Scholarship Level: {prediction['predicted_scholarship']}\n")
            self.prediction_text.insert('end', f"  • 奖学金置信度 Scholarship Confidence: {prediction['scholarship_confidence']:.1f}%\n\n")
            
            risk_color = "🔴" if prediction['at_risk_probability'] > 50 else "🟡" if prediction['at_risk_probability'] > 30 else "🟢"
            self.prediction_text.insert('end', f"⚠️  风险概率 AT-RISK PROBABILITY: {risk_color} {prediction['at_risk_probability']:.1f}%\n\n")
            
            self.prediction_text.insert('end', "💡 建议 RECOMMENDATIONS:\n")
            for i, rec in enumerate(prediction['recommendations'], 1):
                self.prediction_text.insert('end', f"  {i}. {rec}\n")
            
            self.prediction_text.insert('end', "\n" + "="*70 + "\n")
            
        except Exception as e:
            messagebox.showerror("错误 Error", f"预测失败 Prediction failed: {e}")
    
    # ===================== ANALYTICS TAB METHODS =====================
    
    def show_chart(self, chart_type):
        if chart_type != 'importance' and not self.current_results:
            messagebox.showwarning("警告 Warning", "请先评估学生\nPlease evaluate students first")
            return
        
        if chart_type == 'importance' and not self.ml_predictor.is_trained:
            messagebox.showwarning("警告 Warning", "请先训练ML模型\nPlease train ML models first")
            return
        
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(14, 7), dpi=100, facecolor=self.NJUPT_WHITE)
        ax = fig.add_subplot(111)
        
        # Set NJUPT colors for plots
        ax.set_facecolor('#F8F9FA')
        
        if chart_type == 'gpa':
            gpas = [r['academic_score'] for r in self.current_results]
            ax.hist(gpas, bins=20, color=self.NJUPT_BLUE, edgecolor=self.NJUPT_WHITE, alpha=0.8)
            ax.set_title('GPA分布 GPA Distribution', fontsize=16, fontweight='bold', color=self.NJUPT_BLUE, pad=20)
            ax.set_xlabel('学术分数 Academic Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('学生人数 Number of Students', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--', color=self.NJUPT_BLUE)
            
        elif chart_type == 'scholarship':
            scholarships = [r['recommended_scholarship'] for r in self.current_results]
            from collections import Counter
            counts = Counter(scholarships)
            
            colors = ['#FFD700', '#C0C0C0', '#CD7F32', self.NJUPT_BLUE]
            bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor=self.NJUPT_WHITE, linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_title('奖学金分布 Scholarship Distribution', fontsize=16, fontweight='bold', 
                        color=self.NJUPT_BLUE, pad=20)
            ax.set_xlabel('奖学金级别 Scholarship Level', fontsize=12, fontweight='bold')
            ax.set_ylabel('学生人数 Number of Students', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--', color=self.NJUPT_BLUE)
            
        elif chart_type == 'attendance':
            attendance = [r['attendance_rate'] for r in self.current_results]
            scholarships = [r['recommended_scholarship'] for r in self.current_results]
            
            scholarship_types = list(set(scholarships))
            colors = ['#FFD700', '#C0C0C0', '#CD7F32', self.NJUPT_BLUE]
            
            for i, s_type in enumerate(scholarship_types):
                s_attendance = [a for a, s in zip(attendance, scholarships) if s == s_type]
                ax.scatter([s_type]*len(s_attendance), s_attendance, 
                          color=colors[i % len(colors)], label=s_type, alpha=0.7, s=150,
                          edgecolors=self.NJUPT_WHITE, linewidth=2)
            
            ax.set_title('出勤率vs奖学金级别 Attendance vs Scholarship Level', 
                        fontsize=16, fontweight='bold', color=self.NJUPT_BLUE, pad=20)
            ax.set_xlabel('奖学金级别 Scholarship Level', fontsize=12, fontweight='bold')
            ax.set_ylabel('出勤率 Attendance Rate (%)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, framealpha=0.9)
            ax.grid(alpha=0.3, linestyle='--', color=self.NJUPT_BLUE)
            
        elif chart_type == 'scores':
            names = [r['name'][:10] for r in self.current_results[:10]]  # Top 10
            academic = [r['academic_score'] for r in self.current_results[:10]]
            ability = [r['ability_score'] for r in self.current_results[:10]]
            
            x = np.arange(len(names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, academic, width, label='学术分数 Academic Score', 
                          color=self.NJUPT_BLUE, edgecolor=self.NJUPT_WHITE, linewidth=2)
            bars2 = ax.bar(x + width/2, ability, width, label='能力分数 Ability Score', 
                          color=self.ACCENT_LIGHT, edgecolor=self.NJUPT_WHITE, linewidth=2)
            
            ax.set_title('分数分解(前10名) Score Breakdown (Top 10 Students)', 
                        fontsize=16, fontweight='bold', color=self.NJUPT_BLUE, pad=20)
            ax.set_xlabel('学生 Students', fontsize=12, fontweight='bold')
            ax.set_ylabel('分数 Scores', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend(fontsize=11, framealpha=0.9)
            ax.grid(axis='y', alpha=0.3, linestyle='--', color=self.NJUPT_BLUE)
            
        elif chart_type == 'importance':
            if self.ml_predictor.feature_importance:
                features = list(self.ml_predictor.feature_importance.keys())
                importances = list(self.ml_predictor.feature_importance.values())
                
                sorted_idx = np.argsort(importances)
                features_sorted = [features[i] for i in sorted_idx]
                importances_sorted = [importances[i] for i in sorted_idx]
                
                bars = ax.barh(features_sorted, importances_sorted, 
                              color=self.NJUPT_BLUE, edgecolor=self.NJUPT_WHITE, linewidth=2)
                
                # Gradient effect
                for i, bar in enumerate(bars):
                    bar.set_alpha(0.5 + 0.5 * (i / len(bars)))
                
                ax.set_title('GPA预测特征重要性 Feature Importance for GPA Prediction', 
                           fontsize=16, fontweight='bold', color=self.NJUPT_BLUE, pad=20)
                ax.set_xlabel('重要性 Importance', fontsize=12, fontweight='bold')
                ax.set_ylabel('特征 Feature', fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3, linestyle='--', color=self.NJUPT_BLUE)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

# ===================== MAIN EXECUTION =====================

def main():
    root = tk.Tk()
    app = ScholarshipGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
